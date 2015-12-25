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
#else
#include <common/memory.h>
#include <core/fwd.h>
#include <boost/property_tree/ptree.hpp>
#include <vector>
#endif

namespace caspar {

#ifndef CASPAR_2_1
namespace core {
	struct frame_consumer;
	class parameters;
}
#endif

	namespace replay {

#ifndef CASPAR_2_1
		safe_ptr<core::frame_consumer> create_consumer(const core::parameters& params);
#else
		void describe_consumer(core::help_sink& sink, const core::help_repository& repo);
		spl::shared_ptr<core::frame_consumer> create_consumer(
			const std::vector<std::wstring>& params, core::interaction_sink*);
#endif
}}