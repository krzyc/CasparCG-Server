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

#include <core/mixer/audio/audio_mixer.h>
#include <core/monitor/monitor.h>

#include <common/memory.h>

#include <core/fwd.h>

#include <boost/noncopyable.hpp>

struct AVPacket;
struct AVFormatContext;

namespace caspar { namespace ffmpeg {
	
class audio_decoder : public boost::noncopyable
{
public:
	explicit audio_decoder(class input& input, const core::video_format_desc& format_desc, const std::wstring& channel_layout_spec, int audio_stream_index = 0);
	
	audio_decoder(audio_decoder&& other);
	audio_decoder& operator=(audio_decoder&& other);

	std::shared_ptr<AVFrame> operator()();

	uint32_t nb_frames() const;
	const core::audio_channel_layout& channel_layout() const;
	
	std::wstring print() const;
	
	core::monitor::subject& monitor_output();

private:
	struct impl;
	spl::shared_ptr<impl> impl_;
};

}}