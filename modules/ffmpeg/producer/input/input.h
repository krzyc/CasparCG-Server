/*
* Copyright 2013 Sveriges Television AB http://casparcg.com/
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
*/

#pragma once
#include "../../ffmpeg_params.h"

#include <common/memory/safe_ptr.h>

#include <memory>
#include <string>
#include <cstdint>

#include <boost/noncopyable.hpp>
#include <boost/thread/future.hpp>

struct AVFormatContext;
struct AVPacket;

namespace caspar {

namespace diagnostics {

class graph;

}
	 
namespace ffmpeg {

class input : boost::noncopyable
{
public:
	explicit input(const safe_ptr<diagnostics::graph>& graph, const std::wstring& filename, FFMPEG_Resource resource_type, bool loop, uint32_t start, uint32_t length, bool thumbnail_mode, const ffmpeg_producer_params& vid_params);

	bool try_pop(std::shared_ptr<AVPacket>& packet);
	bool eof() const;

	void loop(bool value);
	bool loop() const;

	boost::unique_future<bool> seek(uint32_t target);

	safe_ptr<AVFormatContext> context();
private:
	struct implementation;
	std::shared_ptr<implementation> impl_;
};

	
}}
