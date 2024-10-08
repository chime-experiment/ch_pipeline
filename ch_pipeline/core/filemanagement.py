"""Tools to copy, move, and delete files via the CHIME database."""

import chimedb.core as db
import chimedb.data_index as di
import peewee as pw

from datetime import datetime


def make_copy_request(
    files: list[int],
    source_node_name: str = None,
    target_group_name: str = "cedar_online",
):
    """Make request to CHIME database for files to be copied.

    By default this function assumes files should be copied from Cedar tape
    storage to Cedar online, checking for each file whether it is present in
    the `cedar_nearline` or `cedar_smallfile` nodes and requesting a copy to
    `cedar_online`.

    Parameters
    ----------
    files
        List of indices (in DB table ArchiveFileCopy) of files to be copied.
    source_node_name
        Name of node in CHIME database from which to copy.
    target_group_name
        Name of storage group in CHIME database to which files will be copied.
        Defaults to `cedar_online`
    """

    if isinstance(files, str):
        files = [files]

    # Get possible source nodes
    if source_node_name:
        offline_node = di.StorageNode.get(name=source_node_name)
        smallfile_node = di.StorageNode.get(name=source_node_name)
    else:
        offline_node = di.StorageNode.get(name="cedar_nearline")
        smallfile_node = di.StorageNode.get(name="cedar_smallfile")

    # Get target group
    target_group = di.StorageGroup.get(name=target_group_name)

    # Establish a read-write database connection
    db.connect(read_write=True)

    nrequests = 0

    for file_ in files:
        # Check if file is present in regular nearline or smallfile node and
        # specify the source node for the file accordingly
        try:
            di.ArchiveFileCopy.get(
                file=file_,
                node=offline_node,
                has_file="Y",
            )
            source_node = offline_node
        except pw.DoesNotExist:
            try:
                di.ArchiveFileCopy.get(
                    file=file_,
                    node=smallfile_node,
                    has_file="Y",
                )
                source_node = smallfile_node
            except pw.DoesNotExist:
                raise ValueError(
                    f"File not found on `f{offline_node}` nor on `f{smallfile_node}`: {file_}"
                )

        try:
            # Check if an activate request already exists. If so,
            # leave alpenhorn alone to do its thing
            di.ArchiveFileCopyRequest.get(
                file=file_,
                group_to=target_group,
                node_from=source_node,
                completed=False,
                cancelled=False,
            )
        except pw.DoesNotExist:
            di.ArchiveFileCopyRequest.insert(
                file=file_,
                group_to=target_group,
                node_from=source_node,
                cancelled=0,
                completed=0,
                n_requests=1,
                timestamp=datetime.now(),
            ).execute()
            nrequests += 1

    return nrequests


def make_remove_request(files: list[str], node_name: str = "cedar_online"):
    """Make request to CHIME database for files to be removed.

    Parameters
    ----------
    files
        Files to be removed.
    node_name
        Name of node in CHIME database from which to remove the files.
        Defaults to `cedar_online`
    """

    online_node = di.StorageNode.get(node_name)

    # Establish a read-write database connection
    db.connect(read_write=True)

    # Request that these files be removed from the online node
    di.ArchiveFileCopy.update(wants_file="N").where(
        di.ArchiveFileCopy.file << files,
        di.ArchiveFileCopy.node == online_node,
    ).execute()
