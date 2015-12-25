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
* Author: Robert Nagy, ronag89@gmail.com
*		  Jan Starzak, jan@ministryofgoodsteps.com
*		  Krzysztof Pyrkosz, pyrkosz@o2.pl
*/

#pragma once

#ifndef CASPAR_2_1
#include <common/memory/safe_ptr.h>
#endif

#include <core/producer/frame_producer.h>

#define		REPLAY_PRODUCER_BUFFER_SIZE		3

namespace caspar {

#ifndef CASPAR_2_1
namespace core {
	struct frame_producer;
	struct frame_factory;
	class parameters;
}
#endif

namespace replay {
#ifndef CASPAR_2_1
	safe_ptr<core::frame_producer> create_producer(const safe_ptr<core::frame_factory>& frame_factory, const core::parameters& params);
#else
	void describe_producer(core::help_sink& sink, const core::help_repository& repo);
	spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies, const std::vector<std::wstring>& params);
#endif
}}