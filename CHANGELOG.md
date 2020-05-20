# [20.5.0](https://github.com/chime-experiment/ch_pipeline/releases/tag/v20.5.0) (2020-05-07)


### Bug Fixes

* change versioneer config to give correct version strings ([33859ea](https://github.com/chime-experiment/ch_pipeline/commit/33859eac0b2623bf87a4b81cc9da34384113ec8d))
* **analysis.beam:** Resolve conflicting tools import. ([#34](https://github.com/chime-experiment/ch_pipeline/issues/34)) ([3b01f1a](https://github.com/chime-experiment/ch_pipeline/commit/3b01f1ad91dc19e6f2d98f0aa24366ebe7419467))
* **analysis.beam:** Save a 26m input that can be used to directly index the input axis ([d47dc9b](https://github.com/chime-experiment/ch_pipeline/commit/d47dc9b4f92ff852e661b6afcb23bb5c3b77ae65))
* **base:** changed incorrect mode for jobscript generation ([08083bc](https://github.com/chime-experiment/ch_pipeline/commit/08083bc16425af25ef3a0c76fccf86c2fa7b269c))
* **DataFlagger:** correct location of DataFlag classes ([15ae96a](https://github.com/chime-experiment/ch_pipeline/commit/15ae96a2dd355fee797cd75aaaa540aa5f53fc80))
* **dataquery:** correct the default node_spoof ([0ce7636](https://github.com/chime-experiment/ch_pipeline/commit/0ce7636b0d4b7840032535dc2446a016ab38ecd8))
* **dataquery:** make `QueryAcquisitions` work as a `task.MPILoggedTask`. ([65e125a](https://github.com/chime-experiment/ch_pipeline/commit/65e125a0db8fd09c5d2f5f7d8a3f89a33e769119))
* **flagging:** convert altitude from Angle to radians ([fd4b4dc](https://github.com/chime-experiment/ch_pipeline/commit/fd4b4dc1ee621f2b9eeec9b3814c7ee3db29dd9f))
* **LoadSetup:** file reading bug ([23ba828](https://github.com/chime-experiment/ch_pipeline/commit/23ba828680dc5190a67e774879df645629e72ef6))
* **MaskCHIMEData:** global slicing outside loop ([#24](https://github.com/chime-experiment/ch_pipeline/issues/24)) ([7259c8f](https://github.com/chime-experiment/ch_pipeline/commit/7259c8f5593a2d005a879bb755427502c3a5ddd7))
* **processing:** preserve order of pending items ([59bc24c](https://github.com/chime-experiment/ch_pipeline/commit/59bc24ceacd11e50929b5794fa6147db1d6abcc6))
* **ProcessingType:** string type issue wehn writing out config ([cad0df6](https://github.com/chime-experiment/ch_pipeline/commit/cad0df6ac82a9b077b5d2a3ff077050cd4563105))
* **RFISensitivityMask:** incorrect time range used for bright sources ([888fd42](https://github.com/chime-experiment/ch_pipeline/commit/888fd42388ae90b86c5fc4021d91971acd04d637))
* **RingMapMaker:** calculate grid properly after feed pos rotation ([e24c6c7](https://github.com/chime-experiment/ch_pipeline/commit/e24c6c78cc15efb46ccae5e1727a63a68db1147c))
* change deprecated import location ([1ee39d5](https://github.com/chime-experiment/ch_pipeline/commit/1ee39d52c529cd12cc47f184dcff60386d227099))
* move from deprecated ch_util module to chimedb.data_index ([1faf9d8](https://github.com/chime-experiment/ch_pipeline/commit/1faf9d81a1bf99fc010a9162057e38ab3c136d9b))
* Replace calls to data_index.connect_database. ([fba93aa](https://github.com/chime-experiment/ch_pipeline/commit/fba93aa042be71465e009f997a1ad0fc3d6e6034))
* **requirements.txt:** change ch_util installation from https to ssh ([c396fc6](https://github.com/chime-experiment/ch_pipeline/commit/c396fc6fbd3973ab4a52be33d11a07e30e0d9627))
* **requirements.txt:** update ch_util repository address ([3fdda52](https://github.com/chime-experiment/ch_pipeline/commit/3fdda5278226e786c9a827ff4c136552782bf96f))
* **sidereal,io:** fixes string problems that arise in python 2. ([d794c59](https://github.com/chime-experiment/ch_pipeline/commit/d794c59db4c1b36157b8f2594eb7ef0a770378cc))
* use `prodstack` where appropriate ([1ab071f](https://github.com/chime-experiment/ch_pipeline/commit/1ab071f141d75fb270fcc7abe228ce053c553f50))
* **RingMapMaker:** Fixed calculation of nfeed ([cc263fe](https://github.com/chime-experiment/ch_pipeline/commit/cc263fe4fe38d0787d1859917f7880aaeb35e56d))
* **RingMapMaker:** Fixed ringmap calculation when single_beam=True ([94f51e5](https://github.com/chime-experiment/ch_pipeline/commit/94f51e58873cd6701dd8bf0fb57a74372f121c19))
* **source_removal:** handle csd of sidereal stacks properly ([#26](https://github.com/chime-experiment/ch_pipeline/issues/26)) ([4d38d4e](https://github.com/chime-experiment/ch_pipeline/commit/4d38d4ee079c612bb55cbd87337eec4b9a870089))
* **timing:** modify condition that determines appropriate timing correction. ([856b475](https://github.com/chime-experiment/ch_pipeline/commit/856b4753e626848e017402962b2718d2c024123a))
* remove unicode_literals for better setuptools support ([a9697f3](https://github.com/chime-experiment/ch_pipeline/commit/a9697f3263554e43f2cbd6f7c494621d1dac890c))
* removed debugging clause that disabled fast read. ([05da0d8](https://github.com/chime-experiment/ch_pipeline/commit/05da0d88601c8ec67925132ced18166134b64a9c))


### Features

* **analysis.timing:** Add option to pass if no correction found. ([a56bf92](https://github.com/chime-experiment/ch_pipeline/commit/a56bf929d1706502258086127e2b3008f5b77198))
* **calibration:** pipeline for generating gains from chimecal acquisitions ([#22](https://github.com/chime-experiment/ch_pipeline/issues/22)) ([44f6e6b](https://github.com/chime-experiment/ch_pipeline/commit/44f6e6b6a9778c10dea3f9ca16d22be32cfa8fe3))
* **daily:** allow intervals to be given as CSD ranges ([89685ca](https://github.com/chime-experiment/ch_pipeline/commit/89685ca4eac914dfedbf1017ca0f146fcdbc8277))
* **daily:** updated config for the daily processing ([ab92f66](https://github.com/chime-experiment/ch_pipeline/commit/ab92f667a8e7d3aff4260995eda5d5420c65758d))
* **flagging:** refactor `DayMask`.  add tasks for flagging sources, sun, moon. ([5ddc30a](https://github.com/chime-experiment/ch_pipeline/commit/5ddc30ad8e789bc576ae46fe0559c66616057636))
* **processing.base:** open a description file when user creates a revision. ([2ea4e2e](https://github.com/chime-experiment/ch_pipeline/commit/2ea4e2e339016847a1a2fa7e1d35d6ed62150404))
* **QueryInputs:** add an option to cache and reuse an inputs query ([b3b3e4c](https://github.com/chime-experiment/ch_pipeline/commit/b3b3e4cb6b2a87faf97568551916c71efb9e8366))
* add versioneer for better version naming ([c5e77ba](https://github.com/chime-experiment/ch_pipeline/commit/c5e77bac0459c2ebce90faedda88e465fc4caa6b))
* **processing.client:** Add option to list all types including those with no data. ([6bcc201](https://github.com/chime-experiment/ch_pipeline/commit/6bcc2018a85cf081437820668b3e5b209944a094))
* added ability to filter out specified CSDs ([1d12572](https://github.com/chime-experiment/ch_pipeline/commit/1d12572ee785089fbebe923df9b5a6530592b02e))
* added option to only generate the central beam. ([e5231fc](https://github.com/chime-experiment/ch_pipeline/commit/e5231fcf7183fa13b528776497a4161c06c809b1))
* python 3 support ([33c07bd](https://github.com/chime-experiment/ch_pipeline/commit/33c07bdcc660bee1d62bbb2ab76abe401a914c9e))
* store virtualenv in revision directory. ([1bf909f](https://github.com/chime-experiment/ch_pipeline/commit/1bf909f44e008123005c77106966fc797e6d418b))
* **RFISensitivityMask:** added CHIME specific version of RFI masking task ([a6f471c](https://github.com/chime-experiment/ch_pipeline/commit/a6f471ce198c0de4160ecf29a507124047ac71ed))
* support for fast reading of andata files. ([a05a129](https://github.com/chime-experiment/ch_pipeline/commit/a05a129f9db828a45f9d2ec5dee16b49ad38e186))
* use a nonzero rotation of the CHIME telescope ([4995318](https://github.com/chime-experiment/ch_pipeline/commit/499531804366008f3463d6c3501ec530cabe4659))
* **processing:** tools for managing the regular processing of data ([309f8f4](https://github.com/chime-experiment/ch_pipeline/commit/309f8f4c336b5074fb2f4470344592d4c4f74f77))
* **QueryDatabase:** query for a range of CSDs ([4890838](https://github.com/chime-experiment/ch_pipeline/commit/489083865fd7b6242a7da2459963676643199659))
* **RingMapMaker:** improve handling of cross-pol to keep Stokes V ([3e7c194](https://github.com/chime-experiment/ch_pipeline/commit/3e7c1941c73b008a0bf17078b78f122c7ff1ae63))
* **sidereal:** capability to specify ranges of RA for mean subtraction. ([ea2d6a0](https://github.com/chime-experiment/ch_pipeline/commit/ea2d6a0ecc47ab4ffd7a9693ba6e0add1731ef44))
* **sidereal:** working versions of `SiderealMean` and `ChangeSiderealMean` ([1ffe86a](https://github.com/chime-experiment/ch_pipeline/commit/1ffe86a9674039ae9cdec16f2af60324781de5ad)), closes [#39](https://github.com/chime-experiment/ch_pipeline/issues/39)
* **source_removal:** initial version of tasks that subtract sources from visibilities ([d004807](https://github.com/chime-experiment/ch_pipeline/commit/d0048072ced50394c331d47131154b1ee788afad))
* **telescope:** add option to mask baselines accodring to total length, ([de3b53d](https://github.com/chime-experiment/ch_pipeline/commit/de3b53dc2c45d0239f433f885d1d54eb0da3173f))
* **telescope:** add polarisation property ([#23](https://github.com/chime-experiment/ch_pipeline/issues/23)) ([e35b58c](https://github.com/chime-experiment/ch_pipeline/commit/e35b58cec718e5ec50d10d519c8cad4699002955))
* **telescope:** fix some bugs and style errors ([ef9926a](https://github.com/chime-experiment/ch_pipeline/commit/ef9926ad94fb181d18d894b343bb0757ca82cad6))
* **timing:** pipeline tasks for constructing timing correction and applying to data. ([fe13007](https://github.com/chime-experiment/ch_pipeline/commit/fe130073d6d784384a0481c683b1e31bdb35a1f5)), closes [#33](https://github.com/chime-experiment/ch_pipeline/issues/33)
* **timingerrors.py:** simulate timing distribution errors ([f8437e4](https://github.com/chime-experiment/ch_pipeline/commit/f8437e4f7aaf8979c7f761d7d004dc91e3abe2b3)), closes [#24](https://github.com/chime-experiment/ch_pipeline/issues/24)
* a task that scans for existing CSDs to stop repeat processing ([0285c9c](https://github.com/chime-experiment/ch_pipeline/commit/0285c9cc848f10216e32b89d513e2d87ddb8bb36))


### Performance Improvements

* **RingMap:** change order of axes for faster IO ([0aada0c](https://github.com/chime-experiment/ch_pipeline/commit/0aada0cf377da9965711fe14fc6c4f462c61f3e8))
* **RingMapMaker:** speed up by resolving datasets once ([364cab3](https://github.com/chime-experiment/ch_pipeline/commit/364cab3af17d790759106321badf850357b302d2))


### BREAKING CHANGES

* **RingMap:** the ring map container axis order has changed

