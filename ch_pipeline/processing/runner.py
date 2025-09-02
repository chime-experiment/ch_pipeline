"""Mini daemon for running pipeline jobs."""

import asyncio
import atexit
import getpass
import json
import logging
import traceback
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path

import chimedb.core as db
from apscheduler.schedulers.background import BackgroundScheduler

from ch_pipeline.processing import base, client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_default_config = {
    "update_minutes": 60,
    "max_jobs_number": 3,
    "min_fairshare_number": 1.2,
    "run_indefinitely": False,
}
GLOBAL_CONFIG = {"user": None}

# True if this is the main process running the server
_is_main_process = False

# Store existing processing revisions in the global space
PROCESSING_REVISIONS = {}
# Global scheduler to run tasks in their own threads
SCHEDULER = None


@lru_cache(maxsize=10)
def _format_revision(revision: base.ProcessingType):
    return f"{revision.type_name}:{revision.revision}"


def setup(revision: base.ProcessingType, config: dict = {}):
    """Set up a revision and update the global config."""
    # Check the socket. If it already exists, assume that the
    # pipeline service is running and we just have to submit
    # another job to it. Otherwise, set up the server
    server_is_running = SERVER.check_or_make_server(revision)
    revision_name = _format_revision(revision)

    data = {
        "revision": revision_name,
        "config": config,
    }

    if server_is_running:
        # The server is already running. Submit the job and return
        logger.info("A server is already running.")
        response = send({"route": "/add", "data": data})
        if response.get("result"):
            logger.info(f"Revision {revision_name} was submitted.")
        else:
            logger.warning(
                f"An error occured trying to submit revision "
                f"{revision_name}.\n{response.get('msg')}"
            )
    else:
        # We've just made the server, so register this job to be
        # submitted once the server starts
        global _is_main_process
        _is_main_process = True
        SERVER.queue(partial(add_job, data))


def run():
    """Run the pipeline."""
    if not _is_main_process:
        # This is not the main pipeline runner, so we don't
        # want to run the server here
        return

    logger.info("Starting the server...")
    # Start the background pipeline runner
    SCHEDULER.start()
    # Run the server
    asyncio.run(SERVER.run())

    logger.info("\n\nKilled pipeline service...")
    logger.info("Running cleanup...")


def send(request):
    """Send a request to the server."""
    if SERVER._socket is None:
        # Try to extract the revision from the request
        package = json.loads(request).get("data")
        if isinstance(package, dict):
            revision_name = package.get("revision")
            if revision_name is not None:
                # Load this revision and set the socket
                revision = client.PRev().convert(revision_name, None, None)
                SERVER._socket = Path(revision.root_path) / ".socket"

    if SERVER._socket is None:
        logger.warning("Can't send request - no socket was set.")
        return None

    return asyncio.run(SERVER._send_request(request))


# ------------------------------
# Server tasks
# ------------------------------


class _AsyncServer:
    def __init__(self):
        self._socket = None
        self._routes = {}
        self._startup_tasks = []
        self._server = None
        self._timeout = 10

    @property
    def is_running(self):
        """Is the server running?"""
        return self._server is not None and self._server.is_serving()

    def route(self, route_):
        """Decorator to route functions."""

        def _route(func):
            self._routes[route_] = func
            return func

        return _route

    def queue(self, func):
        """Add a task to run after the server starts.

        Parameters
        ----------
        func : function
            The function to call
        """
        self._startup_tasks.append(func)
        logger.debug(f"Added function {func} to startup queue.")

    def check_or_make_server(self, revision):
        """Set up a server on the given socket.

        If a server is already running, return.

        Parameters
        ----------
        revision
            revision (given as <type>:<rev>) for which to run a server.

        Returns
        -------
        server_is_running
            True if the server is already running
            False if a new server has been started
        """
        socket = Path(revision.root_path) / ".socket"
        # Check the socket. If it already exists, assume that the
        # pipeline service is running. Otherwise, set up the server
        if socket.exists():
            if not socket.is_socket():
                # Something is broken
                raise FileExistsError(
                    "Socket path already exists, but it isn't a socket."
                )
            # Check that the server can be reached
            response = send({"route": "\ping", "data": {}})
            if response is None or not response.get("result"):
                raise FileExistsError("Socket exists but is not responsive.")
            # The server is already running
            return True

        self._socket = socket

        # Try to figure out the current user if it isn't set
        # This will only run the first time the runner is started
        if GLOBAL_CONFIG["user"] is None:
            user = getpass.getuser()
            logger.info(f"Running as user {user}")
            GLOBAL_CONFIG["user"] = user

        # Unlink the socket on close
        atexit.register(self._socket.unlink)

        # Set up the scheduler
        global SCHEDULER
        SCHEDULER = BackgroundScheduler(daemon=True)
        atexit.register(partial(SCHEDULER.shutdown, wait=True))

        return False

    async def _handler(self, reader, writer):
        """Server request handler."""
        try:
            request = await asyncio.wait_for(reader.readline(), timeout=self._timeout)
        except asyncio.TimeoutError:
            return
        request = json.loads(request.decode("utf-8"))
        response = {"result": True}

        # Get the route that's being requested and handle
        # accordingly if the request was bad
        if "route" not in request or "data" not in request:
            response["result"] = False
            response["msg"] = "Bad request"
        else:
            func = self._routes.get(request["route"])
            if func is None:
                response["result"] = False
                response["msg"] = "Bad route"
            else:
                # Get whatever is set at this route
                try:
                    ret = await func(request["data"])
                except:  # noqa: E722
                    response["result"] = False
                    response["msg"] = traceback.format_exc()
                else:
                    # If there was a response, use it. Otherwise we'll
                    # fall back to the default and assume that things
                    # went fine
                    if ret is not None:
                        response = ret

        if isinstance(response, dict):
            response = json.dumps(response)

        writer.write((response + "\n").encode("utf-8"))
        writer.write_eof()
        try:
            await asyncio.wait_for(writer.drain(), timeout=self._timeout)
        except ConnectionResetError:
            logger.debug("Connection closed by peer. No response was sent.")
        except asyncio.TimeoutError:
            logger.warning("Connection timed out sending reply.")

        writer.close()

    async def run(self):
        """Run the server and any other startup tasks."""

        async def _wait_for_server(func):
            # Wait until the server is started to execute
            while not self.is_running:
                await asyncio.sleep(0.5)
            # Wait a bit longer for good measure
            await asyncio.sleep(1)
            await func()
            logger.debug(f"Completed startup task {func}")

        async def _run_server():
            """Make an async server on a unix socket."""
            self._server = await asyncio.start_unix_server(
                self._handler, path=str(self._socket)
            )
            async with self._server:
                await self._server.serve_forever()

        self.tasks = [
            *(_wait_for_server(t) for t in self._startup_tasks),
            _run_server(),
        ]

        return await asyncio.gather(*self.tasks, return_exceptions=True)

    async def close(self):
        if self.is_running:
            self._server.close()
            await self._server.wait_closed()

    async def _send_request(self, data: dict):
        """Send a request to the server."""
        if self._socket is None:
            return "ERROR: No socket exists."
        if "route" not in data:
            return "ERROR: Invalid request. No route"

        if "data" not in data:
            # This is just a request to do something
            data["data"] = {}

        # Send the data
        data = json.dumps(data) + "\n"

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(path=str(self._socket)),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            return None
        # Send the request
        writer.write(data.encode("utf-8"))
        writer.write_eof()
        try:
            await asyncio.wait_for(writer.drain(), timeout=self._timeout)
        except asyncio.TimeoutError:
            writer.close()
            return "ERROR: timed out when writing to server."
        # Wait for a reply
        try:
            response = await asyncio.wait_for(reader.readline(), timeout=self._timeout)
        except asyncio.TimeoutError:
            return "ERROR: timed out waiting for server response."
        response = response.decode("utf-8")
        try:
            response = json.loads(response)
        except Exception:  # noqa: BLE001
            response = None

        return response


# Create a server instance on import
SERVER = _AsyncServer()


# ------------------------------
# Request routes
# ------------------------------


@SERVER.route("/ping")
async def ping(request: dict):
    """Reply to a ping request."""
    return json.dumps({"result": True})


@SERVER.route("/finish")
async def finish(request: dict):
    """Finish a revision and close server if nothing left."""
    revision_name = request.get("revision")
    # response = _default_response.copy()
    revision = PROCESSING_REVISIONS.get(revision_name)

    if revision is None:
        logger.info(f"No revision {revision_name}")
        return json.dumps({"result": False, "msg": f"No revision {revision_name}"})

    if GLOBAL_CONFIG[revision_name]["run_indefinitely"]:
        # Just return, this revision should stick around
        return json.dumps({"result": True})

    SCHEDULER.remove_job(f"runner_{revision_name}")
    PROCESSING_REVISIONS.pop(revision_name, None)

    logger.info(
        f"Done revision {revision_name}. {len(PROCESSING_REVISIONS)} processing "
        f"revision(s) remaining: {list(PROCESSING_REVISIONS.keys())}"
    )

    if not PROCESSING_REVISIONS:
        logger.info("Nothing left to process. Shutting down server.")
        await SERVER.close()

    return json.dumps({"result": True})


@SERVER.route("/metrics")
async def metrics(request: dict):
    """Get metrics about a revision."""
    revision_name = request.get("revision")

    # Get the actual revision object
    revision = PROCESSING_REVISIONS.get(revision_name)

    # The requested revision doesn't exist, so don't do anything
    if revision is None:
        logger.info(f"No revision {revision_name}")
        return json.dumps({"result": False, "msg": None})

    return json.dumps(
        {"result": True, "msg": revision.status(user=GLOBAL_CONFIG["user"])}
    )


@SERVER.route("/add")
async def add_job(request: dict):
    """Add a processing job to the pipeline runner."""
    revision_name = request.get("revision")
    config = request.get("config", {})

    # Load this revision
    revision = client.PRev().convert(revision_name, None, None)

    if revision_name in PROCESSING_REVISIONS:
        logger.info(f"Revision {revision_name} already exists.")
        return json.dumps(
            {"result": True, "msg": f"Revision {revision_name} already exists."}
        )

    # Update the config with any revision-specific items
    GLOBAL_CONFIG[revision_name] = _default_config.copy()
    try:
        GLOBAL_CONFIG[revision_name].update(revision.service_config)
    except AttributeError:
        pass

    # Override with any manually set parameters
    GLOBAL_CONFIG[revision_name].update(config)

    # Add this revision to the global revisions
    PROCESSING_REVISIONS[revision_name] = revision

    # Schedule this revision
    SCHEDULER.add_job(
        partial(_run_pipeline, revision),
        "interval",
        minutes=GLOBAL_CONFIG[revision_name]["update_minutes"],
        next_run_time=datetime.now().astimezone(),  # has to be timezone-aware
        id=f"runner_{revision_name}",
        name=f"pipeline_{revision_name}",
    )
    logger.info(f"Job {revision_name} was added.")
    # Clean up any extra files when the server shuts down
    atexit.register(partial(_update_files, revision, retrieve=False))

    return json.dumps({"result": True})


# ------------------------------
# Pipeline tasks
# ------------------------------


def _run_pipeline(revision):
    """Run all the pipeline jobs in a fixed order."""
    _generate(revision)
    _update_files(revision)
    _check_finished(revision)


@db.atomic
def _generate(revision):
    """Submit some jobs to the queue."""
    revision_name = _format_revision(revision)

    fs = base.slurm_fairshare("rpp-chime_cpu")

    priority_only = False
    fairshare_ = GLOBAL_CONFIG[revision_name]["min_fairshare_number"]

    if not isinstance(fairshare_, float):
        raise ValueError(
            f"Invalid fairshare threshold. Expected `float`, got `{type(fairshare_)}`."
        )

    # Get a differnet priority fairshare, if it exists
    priority_fairshare_ = GLOBAL_CONFIG[revision_name].get(
        "min_fairshare_number_priority", fairshare_
    )

    if fairshare_ > fs[0]:
        if priority_fairshare_ > fs[0]:
            logger.info(
                f"Current fairshare {fs[0]} is lower than priority threshold "
                f"{priority_fairshare_}. Skipping all."
            )
            return
        logger.info(
            f"Current fairshare {fs[0]} is lower than threshold "
            f"{fairshare_}. Skipping non-priority."
        )
        priority_only = True

    number_in_queue, number_running = (len(l) for l in revision.queued())
    number_to_submit = max(
        GLOBAL_CONFIG[revision_name]["max_jobs_number"]
        - number_in_queue
        - number_running,
        0,
    )
    logger.info(
        f"Generating {number_to_submit} jobs ({number_in_queue} jobs already queued)."
    )
    revision.generate(
        max=number_to_submit,
        submit=True,
        priority_only=priority_only,
        check_failed=True,
    )


@db.atomic
def _update_files(revision, *args, **kwargs):
    """Run the file update method for a revision."""
    kwargs.update({"user": GLOBAL_CONFIG["user"]})
    revision.update_files(*args, **kwargs)


def _check_finished(revision):
    """Check if this revision is finished."""
    revision_name = _format_revision(revision)

    if GLOBAL_CONFIG[revision_name]["run_indefinitely"]:
        return

    remaining = revision.status(user=GLOBAL_CONFIG["user"])["not_yet_submitted"]
    if len(remaining) == 0:
        # Send a message to the server that this revision is done
        send({"route": "/finish", "data": {"revision": revision_name}})
