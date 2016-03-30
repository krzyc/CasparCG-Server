/*
* Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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

#include <common/memory.h>
#include <common/forward.h>

#include <core/monitor/monitor.h>

#include <boost/noncopyable.hpp>
#include <boost/rational.hpp>

struct AVFormatContext;
struct AVFrame;
struct AVPacket;

namespace caspar { namespace ffmpeg {

class video_decoder : public boost::noncopyable
{
public:
	explicit video_decoder(class input& input, bool single_threaded);
	
	video_decoder(video_decoder&& other);
	video_decoder& operator=(video_decoder&& other);

	std::shared_ptr<AVFrame> operator()();
	
	int	 width() const;
	int	 height() const;
	bool is_progressive() const;
	uint32_t file_frame_number() const;
	boost::rational<int> framerate() const;

	uint32_t nb_frames() const;

	std::wstring print() const;
		
	core::monitor::subject& monitor_output();

private:
	struct impl;
	spl::shared_ptr<impl> impl_;
};

}}