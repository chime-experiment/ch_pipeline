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

    def get_auto_ids(data, inputmap):
        
        auto_stack_id = []
        ind_XX=[]
        ind_YY=[]
        pol_ind=0
        
        for pp, (this_prod, this_conj) in enumerate(data.index_map['stack']):
            bb, aa = data.index_map['prod'][this_prod]
            if aa==bb:
                auto_stack_id.append(pp)
                inp_aa = inputmap[aa]
                inp_bb = inputmap[bb]
                if tools.is_array_x(inp_aa) and tools.is_array_x(inp_bb):
                    ind_XX.append(pol_ind)
                elif tools.is_array_y(inp_aa) and tools.is_array_y(inp_bb):
                    ind_YY.append(pol_ind)
                pol_ind+=1

        return auto_stack_id,ind_XX,ind_YY 

    def live_feeds(cyl, pol, feedmap):

	start_index_cyl = 2*(ord(cyl)-ord('A'))
	start_index_pol = 0 if (pol=='YY') else 1
	index = start_index_cyl + start_index_pol
	sel_ind = range(index*feed_qnt, (index+1)*(feed_qnt))
	good_feeds = np.sum(feedmap[sel_ind], axis=0)

	return np.asarray(good_feeds)

    def stack_weights(self, data):

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

    return np.sqrt(var).T #Same shape as that of autocorrelation
    
    def stack_auto_noise(self, data):
	#Relative cylinder weightings and intercylinder baselines
	N_XX   = 0
	N_YY   = 0
	Nbl_XX = 0
	Nbl_YY = 0	

	f = data_index.Finder(node_spoof = {"gong" : "/mnt/gong/archive"}) 
	f.set_time_range(start_time, stop_time)
	f.accept_all_global_flags()
	f.only_corr()
	f.filter_acqs((data_index.ArchiveInst.name == 'chimestack'))
	file_list = f.get_results()
	all_files = file_list[0][0]
	if not all_files:
	    raise IndexError()

	inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(start_time),
                                               correlator='chime')
	data = andata.CorrData.from_acq_h5(all_files[0], start=0, stop=1,
                                           datasets=['reverse_map', 'flags/inputs'],
                                           apply_gain=False, renormalize=False)

	auto_stack_id,ind_XX,ind_YY = get_auto_ids(data, inputmap)
	
        freq_sel = slice(fstart, fstop)

	auto_data = andata.CorrData.from_acq_h5(all_files, freq_sel=freq_sel,
                                     datasets=['vis', 'flags/vis_weight','flags/frac_lost','flags/inputs'],
                                     apply_gain=False, renormalize=False,\
                                     stack_sel = auto_stack_id)

	bflag = (auto_data.weight.data == 0.0)
	bvis  = auto_data.vis.data
	bfrac = (auto_data.flags['frac_lost'].data)
	binp  = auto_data.flags['inputs']

	live_feed_cyl_XX=np.zeros((ncyl, len(auto_data.time)))
	live_feed_cyl_YY=np.zeros((ncyl, len(auto_data.time)))

	for i in range(0,ncyl):
	    live_feed_cyl_XX[i]=live_feeds(chr(i+ord('A')),'XX',binp)
	    live_feed_cyl_YY[i]=live_feeds(chr(i+ord('A')),'YY',binp)
	live_feed_frac_XX = live_feed_cyl_XX/float(feed_qnt)
	live_feed_frac_YY = live_feed_cyl_YY/float(feed_qnt)

	vis_avg_XX = np.zeros((nfreq,len(auto_data.time)), dtype='complex64')
	vis_avg_YY = np.zeros((nfreq,len(auto_data.time)), dtype='complex64')

	for i in ind_XX:
	    for j in ind_XX:
		if i>=j: #Avoid double counting
		    continue 
		n_feeds_cyl_XX=live_feeds(chr(i+ord('A')),'XX',binp)*live_feeds(chr(j+ord('A')),'XX',binp)
		N_XX += n_feeds_cyl_XX/feed_qnt**2 #(ntime)
		vis_avg_XX +=  bvis[:,i,:]*bvis[:,j,:]*n_feeds_cyl_XX/feed_qnt**2 #(nfreq,ntime) * (ntime)
		Nbl_XX += n_feeds_cyl_XX
		
	for i in ind_YY:
	    for j in ind_YY:
		if i>=j: #Avoid double counting
		    continue 
		n_feeds_cyl_YY = live_feeds(chr(i+ord('A')-len(ind_XX)),'YY',binp)*live_feeds(chr(j+ord('A')-len(ind_XX)),'YY',binp)
		N_YY += n_feeds_cyl_YY/feed_qnt**2
		vis_avg_YY +=  bvis[:,i,:]*bvis[:,j,:]*n_feeds_cyl_YY/feed_qnt**2
		Nbl_YY += n_feeds_cyl_YY

	tint = np.median(np.diff(auto_data.time)) #there is a slight different in integrations at micro seconds, ignoring it

	n_b_tau_XX = band/nfreq * tint* Nbl_XX * (1-bfrac)
	n_b_tau_YY = band/nfreq * tint* Nbl_YY * (1-bfrac)

	vis_avg_XX_norm=np.sqrt(vis_avg_XX/(N_XX*n_b_tau_XX))
	vis_avg_YY_norm=np.sqrt(vis_avg_YY/(N_YY*n_b_tau_YY))

	ratio_XX=np.asarray((arr_vis_0).T/(vis_avg_XX_norm), dtype='float64')
	ratio_YY=np.asarray((arr_vis_1).T/(vis_avg_YY_norm), dtype='float64')


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
