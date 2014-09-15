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
*/

#include "replay_consumer.h"

#include <common/concurrency/executor.h>
#include <common/exception/exceptions.h>
#include <common/env.h>
#include <common/log/log.h>
#include <common/utility/string.h>
#include <common/concurrency/future_util.h>
#include <common/diagnostics/graph.h>
#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <core/parameters/parameters.h>
#include <core/consumer/frame_consumer.h>
#include <core/video_format.h>
#include <core/mixer/read_frame.h>
#include <core/monitor/monitor.h>

#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>

#include <tbb/concurrent_queue.h>

#include "../util/frame_operations.h"
#include "../util/file_operations.h"

#include <FreeImage.h>
#include <setjmp.h>
#include <vector>

#include <Windows.h>

namespace caspar { namespace replay {
	
struct replay_consumer : public core::frame_consumer
{
	// core::monitor::subject					monitor_subject_;

	core::video_format_desc					format_desc_;
	std::wstring							filename_;
	tbb::atomic<uint64_t>					framenum_;
	std::int16_t							quality_;
	chroma_subsampling						subsampling_;
	//boost::mutex*							file_mutex_;
	mjpeg_file_handle						output_file_;
	mjpeg_file_handle						output_index_file_;
	bool									file_open_;
	executor								encode_executor_;
	const safe_ptr<diagnostics::graph>		graph_;
	mjpeg_process_mode						mode_;
	boost::posix_time::ptime				start_timecode_;

	int										audio_channels_;

	tbb::atomic<int64_t>					current_encoding_delay_;


#define REPLAY_FRAME_BUFFER					32
#define REPLAY_JPEG_QUALITY					90
#define REPLAY_JPEG_SUBSAMPLING				Y422

public:

	// frame_consumer

	replay_consumer(const std::wstring& filename, const short quality, const chroma_subsampling subsampling, int audio_channels)
		: filename_(filename)
		, quality_(quality)
		, subsampling_(subsampling)
		, encode_executor_(print())
	{
		framenum_ = 0;
		current_encoding_delay_ = 0;
		audio_channels_ = audio_channels;

		encode_executor_.set_capacity(REPLAY_FRAME_BUFFER);

		graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
		graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
		graph_->set_color("buffered-video", diagnostics::color(0.1f, 0.1f, 0.8f));
		graph_->set_text(print());
		diagnostics::register_graph(graph_);

		//file_mutex_ = new boost::mutex();
		file_open_ = false;
	}

	virtual void initialize(const core::video_format_desc& format_desc, const core::channel_layout& audio_channel_layout, int)
	{
		format_desc_ = format_desc;

		output_file_ = safe_fopen((env::media_folder() + filename_ + L".MAV").c_str(), GENERIC_WRITE, FILE_SHARE_READ);
		if (output_file_ == NULL)
		{
			CASPAR_LOG(error) << print() <<  L" Can't open file " << filename_ << L".MAV for writing";
			return;
		}
		else
		{
			output_index_file_ = safe_fopen((env::media_folder() + filename_ + L".IDX").c_str(), GENERIC_WRITE, FILE_SHARE_READ);
			if (output_index_file_ == NULL)
			{
				CASPAR_LOG(error) << print() <<  L" Can't open index file " << filename_ << L".IDX for writing";
				safe_fclose(output_file_);
				return;
			}
			else
			{
				file_open_ = true;
			}
		}

		if (format_desc.field_mode == caspar::core::field_mode::progressive)
		{
			mode_ = PROGRESSIVE;
		}
		else if (format_desc.field_mode == caspar::core::field_mode::upper)
		{
			mode_ = UPPER;
		}
		else if (format_desc.field_mode == caspar::core::field_mode::lower)
		{
			mode_ = LOWER;
		}

		start_timecode_ = boost::posix_time::microsec_clock::universal_time();

		write_index_header(output_index_file_, &format_desc, start_timecode_, audio_channels_);
	}

#pragma warning(disable: 4701)
	void encode_video_frame(core::read_frame& frame)
	{
		auto format_desc = format_desc_;
		auto out_file = output_file_;
		auto idx_file = output_index_file_;
		auto quality = quality_;
		auto subsampling = subsampling_;
		
		long long written = 0;

		switch (mode_)
		{
		case PROGRESSIVE:
			written = write_frame(out_file, format_desc.width, format_desc.height, frame.image_data().begin(), quality, PROGRESSIVE, subsampling, frame.audio_data().begin(), frame.audio_data().size());
			write_index(idx_file, written);
			break;
		case UPPER:
			written = write_frame(out_file, format_desc.width, format_desc.height / 2, frame.image_data().begin(), quality, UPPER, subsampling, frame.audio_data().begin(), (frame.audio_data().size()) / 2);
			write_index(idx_file, written);
			written = write_frame(out_file, format_desc.width, format_desc.height / 2, frame.image_data().begin(), quality, LOWER, subsampling, frame.audio_data().begin() + (frame.audio_data().size() / 2), (frame.audio_data().size()) / 2);
			write_index(idx_file, written);
			break;
		case LOWER:
			written = write_frame(out_file, format_desc.width, format_desc.height / 2, frame.image_data().begin(), quality, LOWER, subsampling, frame.audio_data().begin(), (frame.audio_data().size()) / 2);
			write_index(idx_file, written);
			written = write_frame(out_file, format_desc.width, format_desc.height / 2, frame.image_data().begin(), quality, UPPER, subsampling, frame.audio_data().begin() + (frame.audio_data().size() / 2), (frame.audio_data().size()) / 2);
			write_index(idx_file, written);
			break;
		}

		++framenum_;
	}
#pragma warning(default: 4701)

	bool ready_for_frame()
	{
		return encode_executor_.size() < encode_executor_.capacity();
	}

	void mark_dropped()
	{
		graph_->set_tag("dropped-frame");
	}

	virtual boost::unique_future<bool> send(const safe_ptr<core::read_frame>& frame) override
	{				
		if (file_open_)
		{
			if (ready_for_frame())
			{
				encode_executor_.begin_invoke([=]
				{		
					boost::timer frame_timer;

					encode_video_frame(*frame);
			
					graph_->set_text(print());
					graph_->set_value("frame-time", frame_timer.elapsed()*0.5*format_desc_.fps);

					current_encoding_delay_ = frame->get_age_millis();

					/* monitor_subject_	<< core::monitor::message("/profiler/time")		% frame_timer.elapsed() % (1.0/format_desc_.fps);			
								
					monitor_subject_	<< core::monitor::message("/file/time")			% (framenum_ / format_desc_.fps) 
										<< core::monitor::message("/file/frame")		% static_cast<int32_t>(framenum_)
										<< core::monitor::message("/file/fps")			% format_desc_.fps
										<< core::monitor::message("/file/path")			% filename_; */
				});
			}
			else
			{
				mark_dropped();
			}

			graph_->set_value("buffered-video", (double)encode_executor_.size() / (double)encode_executor_.capacity());
		}

		return wrap_as_future(true);
	}

	virtual int64_t presentation_frame_age_millis() const override
	{
		return file_open_ ? this->current_encoding_delay_ : 0;
	}

	virtual bool has_synchronization_clock() const override
	{
		return false;
	}

	virtual size_t buffer_depth() const override
	{
		return 1;
	}

	virtual std::wstring print() const override
	{
		return L"replay_consumer[" + filename_ + L".mav|" + std::to_wstring(framenum_) + L"]";
	}

	virtual boost::property_tree::wptree info() const override
	{
		boost::property_tree::wptree info;
		info.add(L"type", L"replay-consumer");
		info.add(L"filename", filename_ + L".mav");
		info.add(L"start-timecode", boost::posix_time::to_iso_wstring(start_timecode_));
		info.add(L"recording-head", framenum_);
		return info;
	}

	virtual int index() const override
	{
		return 150;
	}

	~replay_consumer()
	{
		encode_executor_.stop();
		encode_executor_.join();

		if (output_file_ != NULL)
			safe_fclose(output_file_);

		if (output_index_file_ != NULL)
			safe_fclose(output_index_file_);

		CASPAR_LOG(info) << print() << L" Successfully Uninitialized.";	
	}
};

safe_ptr<core::frame_consumer> create_consumer(const core::parameters& params)
{
	if(params.size() < 1 || params[0] != L"REPLAY")
		return core::frame_consumer::empty();

	std::wstring filename = L"REPLAY";

	short quality = REPLAY_JPEG_QUALITY;
	chroma_subsampling subsampling = REPLAY_JPEG_SUBSAMPLING;
	

	if (params.size() > 1)
	{
		filename = params[1];

		for (uint16_t i=2; i<params.size(); i++)
		{
			if (params[i] == L"SUBSAMPLING")
			{
				if (params[i+1] == L"444")
				{
					subsampling = Y444;
					i++;
				}
				else if (params[i+1] == L"422")
				{
					subsampling = Y422;
					i++;
				}
				else if (params[i+1] == L"420")
				{
					subsampling = Y420;
					i++;
				}
				else if (params[i+1] == L"411")
				{
					subsampling = Y411;
					i++;
				}
			}
			else if (params[i] == L"QUALITY")
			{
				quality = boost::lexical_cast<short>(params[i+1]);
				i++;
			}
		}
	}

	return make_safe<replay_consumer>(filename, quality, subsampling, 2);
}

}}