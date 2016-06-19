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

#include <common/memory.h>

#include <boost/rational.hpp>
#include <boost/noncopyable.hpp>

#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
extern "C"
{
#include <libavutil/samplefmt.h>
}
#if defined(_MSC_VER)
#pragma warning (pop)
#endif

struct AVFrame;

namespace caspar { namespace ffmpeg {

struct audio_input_pad
{
	boost::rational<int>	time_base;
	int						sample_rate;
	AVSampleFormat			sample_fmt;
	std::int64_t			audio_channel_layout;

	audio_input_pad(
			boost::rational<int> time_base,
			int sample_rate,
			AVSampleFormat sample_fmt,
			std::int64_t audio_channel_layout)
		: time_base(std::move(time_base))
		, sample_rate(sample_rate)
		, sample_fmt(sample_fmt)
		, audio_channel_layout(audio_channel_layout)
	{
	}
};

struct audio_output_pad
{
	std::vector<int>			sample_rates;
	std::vector<AVSampleFormat>	sample_fmts;
	std::vector<std::int64_t>	audio_channel_layouts;

	audio_output_pad(
			std::vector<int> sample_rates,
			std::vector<AVSampleFormat> sample_fmts,
			std::vector<std::int64_t> audio_channel_layouts)
		: sample_rates(std::move(sample_rates))
		, sample_fmts(std::move(sample_fmts))
		, audio_channel_layouts(std::move(audio_channel_layouts))
	{
	}
};

class audio_filter : boost::noncopyable
{
public:
	audio_filter(
			std::vector<audio_input_pad> input_pads,
			std::vector<audio_output_pad> output_pads,
			const std::string& filtergraph);
	audio_filter(audio_filter&& other);
	audio_filter& operator=(audio_filter&& other);

	void push(int input_pad_id, const std::shared_ptr<AVFrame>& frame);
	std::shared_ptr<AVFrame> poll(int output_pad_id);
	std::vector<spl::shared_ptr<AVFrame>> poll_all(int output_pad_id);

	std::wstring filter_str() const;
private:
	struct implementation;
	spl::shared_ptr<implementation> impl_;
};

}}
