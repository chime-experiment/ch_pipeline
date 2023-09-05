from datetime import datetime
import shutil
import logging
import getpass
import json
from functools import partial, lru_cache
from pathlib import Path
import atexit
import asyncio
import traceback

from apscheduler.schedulers.background import BackgroundScheduler

import chimedb.core as db

from ch_pipeline.processing import base
from ch_pipeline.processing import client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_default_config = {
    "update_minutes": 60,
    "max_jobs_number": 3,
    "min_fairshare_number": 1.2,
    "run_indefinitely": False,
}
GLOBAL_CONFIG = {"user": None}

PROCESSING_REVISIONS = {}
SCHEDULER = None
SOCKET = None
routes = {}

_main_process = False
_is_running = False
_startup_tasks = []


@lru_cache(maxsize=10)
def _format_revision(revision: base.ProcessingType):
    return f"{revision.type_name}:{revision.revision}"


def setup(revision: base.ProcessingType, config: dict = {}):
    """Set up a revision and update the global config."""

    # Check the socket. If it already exists, assume that the
    # pipeline service is running and we just have to submit
    # another job to it. Otherwise, set up the server
    global SOCKET
    SOCKET = Path(revision.root_path) / ".socket"
    server_is_running = _check_or_make_server()
    revision_name = _format_revision(revision)

    data = {
        "revision": revision_name,
        "config": config,
    }

    if server_is_running:
        # The server is already running. Submit the job and return
        logger.info("A server is already running.")
        response = _send_request({"route": "/add", "data": data}, SOCKET)
        if response.get("result"):
            logger.info(f"Revision {revision_name} was submitted.")
        else:
            logger.warning(
                f"An error occured trying to submit revision "
                f"{revision_name}.\n{response.get('opt')}"
            )
    else:
        # We've just made the server, so register this job to be
        # submitted once the server starts
        global _main_process
        _main_process = True
        _startup_tasks.append(partial(add_job, data))


def run():
    """Run the pipeline."""
    if not _main_process:
        # This is not the main pipeline runner, so we don't
        # want to run the server here
        return

    logger.info("Starting the server...")
    # Start the background pipeline runner
    SCHEDULER.start()
    # Run the server
    try:
        asyncio.run(_run_all())
    except (KeyboardInterrupt, SystemExit):
        pass

    logger.info("\n\nKilled pipeline service...")
    logger.info("Running cleanup...")


# ------------------------------
# Server tasks
# ------------------------------


_default_response = {"result": True, "opt": None}


def _send_request(data: dict, socket: Path):
    """Send a datapacket while blocking.

    This should not be called from the main loop.
    """
    return asyncio.run(_async_send_request(data, socket))


async def _async_send_request(data: dict, socket: Path):
    """Send a request to the server."""
    if socket is None:
        return "ERROR: No socket exists."
    if "route" not in data:
        return "ERROR: Invalid request. No route"

    if "data" not in data:
        # This is just a request for something
        data["data"] = None

    # Send the data
    data = json.dumps(data) + "\n"

    reader, writer = await asyncio.open_unix_connection(path=str(socket))
    writer.write(data.encode("utf-8"))
    writer.write_eof()
    await writer.drain()
    response = await reader.read(-1)

    return json.loads(response.decode("utf-8"))


async def _handler(reader, writer):
    request = await reader.read(-1)
    request = json.loads(request.decode("utf-8"))
    response = _default_response.copy()

    # Get the route that's being requested and handle
    # accordingly if the request was bad
    if "route" not in request or "data" not in request:
        response["result"] = False
        response["opt"] = "Bad request"
    else:
        func = routes.get(request["route"])
        if func is None:
            response["result"] = False
            response["opt"] = "Bad route"
        else:
            # Get whatever is set at this route
            try:
                ret = await func(request["data"])
            except Exception:
                response["result"] = False
                response["opt"] = traceback.format_exc()
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
        await writer.drain()
    except ConnectionResetError:
        logger.debug("Connection closed by peer. No response was sent.")
    writer.close()


async def _run_server():
    """Make an async server on a unix socket."""
    server = await asyncio.start_unix_server(_handler, path=str(SOCKET))
    async with server:
        global _is_running
        _is_running = True
        await server.serve_forever()


async def _run_all():
    """Run the server and any other startup tasks."""

    async def _wait_for_server(func):
        # Wait until the server is started to execute
        while not _is_running:
            await asyncio.sleep(1)
        await func()

    tasks_ = [_wait_for_server(t) for t in _startup_tasks] + [_run_server()]
    return await asyncio.gather(*tasks_)


def _check_or_make_server():
    """Set up the server."""
    # Check the socket. If it already exists, assume that the
    # pipeline service is running. Otherwise, set up the server
    if SOCKET.exists():
        if not SOCKET.is_socket():
            # Something is broken
            raise FileExistsError("Socket path already exists, but it isn't a socket.")
        # The server is already running elsewhere
        return True

    # Try to figure out the current user if it isn't set
    # This will only run the first time the runner is started
    if GLOBAL_CONFIG["user"] is None:
        user = getpass.getuser()
        logger.info(f"Running as user {user}")
        GLOBAL_CONFIG["user"] = user

    # Unlink the socket on close
    atexit.register(SOCKET.unlink)

    # Set up the scheduler
    global SCHEDULER
    SCHEDULER = BackgroundScheduler(daemon=True)
    atexit.register(partial(SCHEDULER.shutdown, wait=True))

    return False


# ------------------------------
# Request routes
# ------------------------------


def route(route_):
    def _route(func):
        routes[route_] = func
        return func

    return _route


@route("/finish")
async def finish(request: dict):
    """Finish a revision and close server if nothing left."""
    revision_name = request.get("revision")
    response = _default_response.copy()
    revision = PROCESSING_REVISIONS.get(revision_name)

    if revision is None:
        logger.info(f"No revision {revision}")
        return json.dumps({"result": False, "opt": None})

    if GLOBAL_CONFIG[revision_name]["run_indefinitely"]:
        # Just return, this revision should stick around
        return json.dumps(response)

    try:
        SCHEDULER.remove_job(f"runner_{revision_name}")
    except Exception:
        logger.warning(f"Error removing job: runner_{revision_name}")
        logger.warning(traceback.format_exc())
    else:
        PROCESSING_REVISIONS.pop(revision_name, None)

    logger.info(
        f"Done revision {revision_name}. {len(PROCESSING_REVISIONS)} processing "
        f"revisions remaining: {list(PROCESSING_REVISIONS.keys())}"
    )

    if not PROCESSING_REVISIONS:
        logger.info("Nothing left to process. Shutting down server.")
        raise SystemExit

    return json.dumps(response)


@route("/metrics")
async def metrics(request: dict):
    """Get metrics about a revision."""
    revision_name = request.get("revision")

    # Get the actual revision object
    revision = PROCESSING_REVISIONS.get(revision_name)

    # The requested revision doesn't exist, so don't do anything
    if revision is None:
        logger.info(f"No revision {revision_name}")
        return json.dumps({"result": False, "opt": None})

    return json.dumps(
        {"result": True, "opt": revision.status(user=GLOBAL_CONFIG["user"])}
    )


@route("/add")
async def add_job(request: dict):
    """Add a processing job to the pipeline runner."""

    revision_name = request.get("revision")
    config = request.get("config", {})
    response = _default_response.copy()

    # Load this revision
    revision = client.PRev().convert(revision_name, None, None)

    if revision_name in PROCESSING_REVISIONS:
        logger.info(f"Revision {revision} already exists.")
        return json.dumps(response)

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
    atexit.register(partial(_update_files, revision, bring_online=False))

    return json.dumps(response)


# ------------------------------
# Pipeline tasks
# ------------------------------


def _run_pipeline(revision):
    """Run all the pipeline jobs in a fixed order."""
    _requeue_failed(revision)
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
        logger.info(
            f"Current fairshare {fs[0]} is lower than threshold "
            f"{fairshare_}. Skipping."
        )
        priority_only = True

    if priority_only and priority_fairshare_ > fs[0]:
        logger.info(
            f"Current fairshare {fs[0]} is lower than priority threshold "
            f"{priority_fairshare_}. Skipping all."
        )
        return

    number_in_queue, number_running = [len(l) for l in revision.queued()]
    number_to_submit = max(
        GLOBAL_CONFIG[revision_name]["max_jobs_number"]
        - number_in_queue
        - number_running,
        0,
    )
    logger.info(
        f"Generating {number_to_submit} jobs ({number_in_queue} jobs already queued)."
    )
    revision.generate(max=number_to_submit, submit=True, priority_only=priority_only)


@db.atomic
def _update_files(revision, *args, **kwargs):
    """Run the file update method for a revision."""

    kwargs.update({"user": GLOBAL_CONFIG["user"]})
    revision.update_files(*args, **kwargs)


def _requeue_failed(revision):
    """Re-queue failed jobs based on certain criteria."""

    failed = revision.failed(user=GLOBAL_CONFIG["user"])
    # These causes of failure are generally a result of a one-off issue
    # with the compute node rather that the data itself, so they should
    # be re-run
    requeue = {"chimedb_error", "time_limit", "mpi_error"}

    for key, tags in failed.items():
        if key not in requeue:
            continue
        # Delete these tags so they get requeued
        for tag in tags:
            path = revision.workdir_path / str(tag)
            try:
                shutil.rmtree(path)
            except Exception:
                logger.warning(f"Could not re-queue job with tag {tag}")
                logger.warning(traceback.format_exc())


def _check_finished(revision):
    """Check if this revision is finished."""

    revision_name = _format_revision(revision)

    if GLOBAL_CONFIG[revision_name]["run_indefinitely"]:
        return

    remaining = revision.status(user=GLOBAL_CONFIG["user"])["not_yet_submitted"]
    if len(remaining) == 0:
        # Send a message to the server that this revision is done
        _send_request({"route": "/finish", "data": _format_revision(revision)}, SOCKET)
