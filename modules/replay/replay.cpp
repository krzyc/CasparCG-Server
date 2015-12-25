/*
* Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
* Copyright (c) 2013 Technical University of Lodz Multimedia Centre <office@cm.p.lodz.pl>
*
* This file is part of CasparCG (www.casparcg.com).
*
* CasparCG is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CasparCG is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
*
* Author: Jan Starzak, jan@ministryofgoodsteps.com
*		  Krzysztof Pyrkosz, pyrkosz@o2.pl
*/

#include "replay.h"

#include "producer/replay_producer.h"
#include "consumer/replay_consumer.h"

#include <setjmp.h>
#include <jpeglib.h>
#include <core/producer/frame_producer.h>
#include <core/consumer/frame_consumer.h>
#ifndef CASPAR_2_1
#include <core/parameters/parameters.h>
#include <common/utility/string.h>
#else
#include <core/module_dependencies.h>
#endif
namespace caspar { namespace replay {

#ifndef JCS_EXTENSIONS
#warning Building against non-turbo libjpeg
#define JCS_EXT_RGB 6
#endif

int jpeg_version = -1;
int jpeg_is_turbo = 0;

void JPEGVersionError(j_common_ptr cinfo)
{
	jpeg_version = cinfo->err->msg_parm.i[0];
}

#ifndef CASPAR_2_1
void init()
#else
void init(core::module_dependencies dependencies)
#endif
{
	jpeg_compress_struct cinfo;
	jpeg_error_mgr error_mgr;
	error_mgr.error_exit = &JPEGVersionError;
	cinfo.err = &error_mgr;

	jpeg_create_compress(&cinfo);
	cinfo.input_components = 3;
	jpeg_set_defaults(&cinfo);
	cinfo.in_color_space = JCS_EXT_RGB;
	try
	{
		jpeg_default_colorspace(&cinfo);
		if (jpeg_version == -1)
		{
			jpeg_is_turbo = 1;
			jpeg_destroy_compress(&cinfo);
		}
	}
	catch (...)
	{
		CASPAR_LOG(error) << "[replay] JPEG lib test exception" << std::endl;
	}
	jpeg_CreateCompress(&cinfo, -1, sizeof(cinfo)); // Pass version = -1 to always force error

	CASPAR_LOG(info) << "[replay] Using libjpeg" << (jpeg_is_turbo == 1 ? "-turbo" : "") << " version " << jpeg_version << std::endl;

#ifndef CASPAR_2_1
	core::register_consumer_factory([](const core::parameters& params){return replay::create_consumer(params);});
	core::register_producer_factory(create_producer);
#else
	dependencies.consumer_registry->register_consumer_factory(L"Replay Consumer", create_consumer, describe_consumer);
	dependencies.producer_registry->register_producer_factory(L"Replay Producer", create_producer, describe_producer);
#endif
}

std::wstring get_version()
{
	return L"0.1-beta";
}

}}