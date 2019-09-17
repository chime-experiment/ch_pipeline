# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

from datetime import datetime
import calendar
from caput import config
from dias.utils.string_converter import str2timedelta
from chimedb import data_index
import sqlite3

import os
import subprocess
import gc

import h5py
import numpy as np

from collections import Counter
from collections import defaultdict

from ch_util import andata
from ch_util import tools
from ch_util import ephemeris
from dias import exception
from dias import __version__ as dias_version_tag
from caput import mpiutil, mpiarray, memh5, config, pipeline
from ch_util import rfi, data_quality, tools, ephemeris, cal_utils, andata
from ch_util import data_index as di

from draco.core import task, io

from ..core import containers

class generate_products(task.SingleTask):
    """Generate time-frequency dataset for weights and autocorrelations

    """
    #Define some parameters to be used. Should mostly be the time

    def process(self, data):

    def stack_dataset(self, data):

        # Get Unix time for the start time for timestamp
        time_tuple = start_time.timetuple()
        start_time_unix = calendar.timegm(time_tuple)
        timestamp0 = start_time_unix

        # Look up inputmap
        inputmap = tools.get_correlator_inputs(
            ephemeris.unix_to_datetime(timestamp0), correlator=self.correlator
        )

        # Read a sample file for getting index map
        file_sample = all_files[0]
        data = andata.CorrData.from_acq_h5(
            file_sample,
            datasets=["reverse_map", "flags/inputs"],
            apply_gain=False,
            renormalize=False,
        )

        # Get baselines
        prod, prodmap, dist, conj, cyl, scale = self.get_baselines(data.index_map, inputmap, data.reverse_map)

        for files in all_files:

            # Load index map and reverse map
            data = andata.CorrData.from_acq_h5(
                files,
                datasets=["reverse_map", "flags/inputs"],
                apply_gain=False,
                renormalize=False,
            )

            flag_ind = data.flags["inputs"]

            # Determine axes
            nfreq = data.nfreq
            nblock = int(np.ceil(nfreq / float(self.nfreq_per_block)))

            # Also used in the output file name and database
            timestamp = data.time
            ntime = timestamp.size

            # Determine groups
            polstr = np.array(sorted(prod.keys()))
            npol = polstr.size

            # Calculate counts
            cnt = np.zeros((data.index_map["stack"].size, ntime), dtype=np.float32)

            if np.any(flag_ind[:]):
                for pp, ss in zip(
                    data.index_map["prod"][:], data.reverse_map["stack"]["stack"][:]
                ):
                    cnt[ss, :] += flag_ind[pp[0], :] * flag_ind[pp[1], :]
            else:
                for ss, val in Counter(
                    data.reverse_map["stack"]["stack"][:]
                ).iteritems():
                    cnt[ss, :] = val

            # Initialize arrays
            var = np.zeros((nfreq, npol, ntime), dtype=np.float32)
            counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

            # Loop over frequency blocks
            for index_0, block_number in enumerate(range(nblock)):

                fstart = block_number * self.nfreq_per_block
                fstop = min((block_number + 1) * self.nfreq_per_block, nfreq)
                freq_sel = slice(fstart, fstop)

                self.logger.debug(
                    "Processing block %d (of %d):  %d - %d"
                    % (block_number + 1, nblock, fstart, fstop)
                )

                bdata = andata.CorrData.from_acq_h5(
                    files,
                    freq_sel=freq_sel,
                    datasets=["flags/vis_weight"],
                    apply_gain=False,
                    renormalize=False,
                )

                bflag = (bdata.weight[:] > 0.0).astype(np.float32)
                bvar = tools.invert_no_zero(bdata.weight[:])

                # Loop over polarizations
                for ii, pol in enumerate(polstr):

                    pvar = bvar[:, prod[pol], :]
                    pflag = bflag[:, prod[pol], :]
                    pcnt = cnt[np.newaxis, prod[pol], :]
                    pscale = scale[pol][np.newaxis, :, np.newaxis]

                    var[freq_sel, ii, :] += np.sum(
                        (pscale * pcnt) ** 2 * pflag * pvar, axis=1
                    )
                    counter[freq_sel, ii, :] += np.sum(pscale * pcnt * pflag, axis=1)

                del bdata
                gc.collect()

            # Normalize
            inv_counter = tools.invert_no_zero(counter)
            var *= inv_counter ** 2

            # Compute metric to be exported
            self.sens.labels(pol="EW").set(
                1.0e6 * np.sqrt(1.0 / np.sum(tools.invert_no_zero(var[:, 0, :])))
            )
            self.sens.labels(pol="NS").set(
                1.0e6 * np.sqrt(1.0 / np.sum(tools.invert_no_zero(var[:, 1, :])))
            )

            # Write to file
            output_file = os.path.join(
                self.write_dir, "%d_%s.h5" % (timestamp[0], self.output_suffix)
            )
            self.logger.info("Writing output file...")
            self.update_data_index(data.time[0], data.time[-1], filename=output_file)

            with h5py.File(output_file, "w") as handler:

                index_map = handler.create_group("index_map")
                index_map.create_dataset("freq", data=data.index_map["freq"][:])
                index_map.create_dataset("pol", data=np.string_(polstr))
                index_map.create_dataset("time", data=data.time)

                dset = handler.create_dataset("rms", data=np.sqrt(var))
                dset.attrs["axis"] = np.array(["freq", "pol", "time"], dtype="S")

                dset = handler.create_dataset("count", data=counter.astype(np.int))
                dset.attrs["axis"] = np.array(["freq", "pol", "time"], dtype="S")

                handler.attrs["instrument_name"] = self.correlator
                handler.attrs["collection_server"] = subprocess.check_output(
                    ["hostname"]
                ).strip()
                handler.attrs["system_user"] = subprocess.check_output(
                    ["id", "-u", "-n"]
                ).strip()
                handler.attrs["git_version_tag"] = dias_version_tag
            self.logger.info("File successfully written out.")

    def get_baselines(self, indexmap, inputmap, reversemap):
        """Return baseline indices for averaging."""
        prod = defaultdict(list)
        prodmap = defaultdict(list)
        dist = defaultdict(list)
        conj = defaultdict(list)
        cyl = defaultdict(list)
        scale = defaultdict(list)

        feedpos = tools.get_feed_positions(inputmap)
        new_stack, new_flags = tools.redefine_stack_index_map(inputmap, indexmap['prod'], indexmap['stack'], reversemap)

        for pp, (this_prod, this_conj) in enumerate(new_stack):
                        
            if new_flags[pp]=='False':
                continue

            if this_conj:
                bb, aa = indexmap["prod"][this_prod]
            else:
                aa, bb = indexmap["prod"][this_prod]

            inp_aa = inputmap[aa]
            inp_bb = inputmap[bb]

            if not tools.is_chime(inp_aa) or not tools.is_chime(inp_bb):
                continue

            if not self.include_auto and (aa == bb):
                continue

            if not self.include_intracyl and (inp_aa.cyl == inp_bb.cyl):
                continue

            this_dist = list(feedpos[aa, :] - feedpos[bb, :])

            if tools.is_array_x(inp_aa) and tools.is_array_x(inp_bb):
                key = "XX"

            elif tools.is_array_y(inp_aa) and tools.is_array_y(inp_bb):
                key = "YY"

            elif not self.include_crosspol:
                continue

            elif tools.is_array_x(inp_aa) and tools.is_array_y(inp_bb):
                key = "XY"

            elif tools.is_array_y(inp_aa) and tools.is_array_x(inp_bb):
                key = "YX"

            else:
                raise RuntimeError("CHIME feeds not polarized.")

            this_cyl = "%s%s" % (self.get_cyl(inp_aa.cyl), self.get_cyl(inp_bb.cyl))
            if self.sep_cyl:
                key = key + "-" + this_cyl

            prod[key].append(pp)
            prodmap[key].append((aa, bb))
            conj[key].append(this_conj)
            dist[key].append(this_dist)
            cyl[key].append(this_cyl)

            if aa == bb:
                scale[key].append(0.5)
            else:
                scale[key].append(1.0)

        for key in prod.keys():
            prod[key] = np.array(prod[key])
            prodmap[key] = np.array(
                prodmap[key], dtype=[("input_a", "<u2"), ("input_b", "<u2")]
            )
            dist[key] = np.array(dist[key])
            conj[key] = np.nonzero(np.ravel(conj[key]))[0]
            cyl[key] = np.array(cyl[key])
            scale[key] = np.array(scale[key])

        tools.change_chime_location(default=True)

        return prod, prodmap, dist, conj, cyl, scale

    def get_cyl(self, cyl_num):
        """Return the cylinfer ID (char)."""
        return chr(cyl_num - self.cyl_start_num + self.cyl_start_char)
