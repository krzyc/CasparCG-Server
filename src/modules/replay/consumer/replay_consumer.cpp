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
*          Jan Starzak, jan@ministryofgoodsteps.com
*          Krzysztof Pyrkosz, pyrkosz@o2.pl
*/

#include "replay_consumer.h"

#include "../util/frame_operations.h"
#include "../util/file_operations.h"

#include <core/frame/frame.h>
#include <common/executor.h>
#include <common/env.h>
#include <common/future.h>
#include <common/diagnostics/graph.h>
#include <common/param.h>
#include <core/consumer/frame_consumer.h>
#include <core/video_format.h>
#include <core/mixer/mixer.h>
#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>
#include <tbb/concurrent_queue.h>
#include <boost/timer.hpp>

namespace caspar { namespace replay {

struct replay_consumer : public core::frame_consumer
{
    core::monitor::state                       state_;
    mutable std::mutex                         state_mutex_;
    core::video_format_desc                    format_desc_;
    std::wstring                               filename_;
    tbb::atomic<uint64_t>                      framenum_;
    std::int16_t                               quality_;
    chroma_subsampling                         subsampling_;
    mjpeg_file_handle                          output_file_;
    mjpeg_file_handle                          output_index_file_;
    bool                                       file_open_;
    executor                                   encode_executor_;
    spl::shared_ptr<diagnostics::graph>        graph_;
    boost::posix_time::ptime                   start_timecode_;

    int                                        audio_channels_;

#define REPLAY_FRAME_BUFFER                    32
#define REPLAY_JPEG_QUALITY                    90
#define REPLAY_JPEG_SUBSAMPLING                Y422

public:

    // frame_consumer

    replay_consumer(const std::wstring& filename, const short quality, const chroma_subsampling subsampling, int audio_channels)
        : filename_(filename)
        , quality_(quality)
        , subsampling_(subsampling)
        , encode_executor_(print())
    {
        framenum_ = 0;
        audio_channels_ = audio_channels;

        encode_executor_.set_capacity(REPLAY_FRAME_BUFFER);

        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_color("buffered-video", diagnostics::color(0.1f, 0.1f, 0.8f));
        graph_->set_text(print());
        diagnostics::register_graph(graph_);

        file_open_ = false;
    }

    virtual void initialize(const core::video_format_desc& format_desc, int)
    {
        format_desc_ = format_desc;

        output_file_ = safe_fopen((env::media_folder() + filename_ + L".mav").c_str(), GENERIC_WRITE, FILE_SHARE_READ);
        if (output_file_ == NULL)
        {
            CASPAR_LOG(error) << print() << L" Can't open file " << filename_ << L".mav for writing";
            return;
        }
        else
        {
            output_index_file_ = safe_fopen((env::media_folder() + filename_ + L".idx").c_str(), GENERIC_WRITE, FILE_SHARE_READ);
            if (output_index_file_ == NULL)
            {
                CASPAR_LOG(error) << print() << L" Can't open index file " << filename_ << L".idx for writing";
                safe_fclose(output_file_);
                return;
            }
            else
            {
                file_open_ = true;
            }
        }

        start_timecode_ = boost::posix_time::microsec_clock::universal_time();

        write_index_header(output_index_file_, &format_desc, start_timecode_, audio_channels_);
    }

#pragma warning(disable: 4701)
    void encode_video_frame(core::const_frame frame)
    {
        auto format_desc = format_desc_;
        auto out_file = output_file_;
        auto idx_file = output_index_file_;
        auto quality = quality_;
        auto subsampling = subsampling_;

        long long written = 0;

        written = write_frame(out_file, format_desc.width, format_desc.height, frame.image_data(0).begin(), quality, PROGRESSIVE, subsampling, frame.audio_data().begin(), (uint32_t)frame.audio_data().size()*4);
        write_index(idx_file, written);

        ++framenum_;
    }
#pragma warning(default: 4701)

    bool ready_for_frame()
    {
        return encode_executor_.size() < encode_executor_.capacity();
    }

    void mark_dropped()
    {
        graph_->set_tag(caspar::diagnostics::tag_severity::WARNING, "dropped-frame");
    }

    std::future<bool> send(core::const_frame frame) override
    {
        if (file_open_)
        {
            if (ready_for_frame())
            {
                encode_executor_.begin_invoke([=]
                {
                    boost::timer frame_timer;

                    encode_video_frame(frame);

                    graph_->set_text(print());
                    graph_->set_value("frame-time", frame_timer.elapsed()*0.5*format_desc_.fps);

                    state_["profiler/time"] = {frame_timer.elapsed(), 1.0 / format_desc_.fps};
                    state_["file/time"] = framenum_ / format_desc_.fps;
                    state_["file/frame"] = static_cast<int32_t>(framenum_);
                    state_["file/fps"] = format_desc_.fps;
                    state_["file/path"] = filename_;
                });
            }
            else
            {
                mark_dropped();
            }

            graph_->set_value("buffered-video", (double)encode_executor_.size() / (double)encode_executor_.capacity());
        }

        return make_ready_future(true);
    }

    virtual bool has_synchronization_clock() const override
    {
        return false;
    }

    virtual std::wstring print() const override
    {
        return L"replay_consumer[" + filename_ + L".mav|" + std::to_wstring(framenum_) + L"]";
    }

    virtual int index() const override
    {
        return 150;
    }

    core::monitor::state state() const override
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    ~replay_consumer()
    {
        if (output_file_ != NULL)
            safe_fclose(output_file_);

        if (output_index_file_ != NULL)
            safe_fclose(output_index_file_);

        CASPAR_LOG(info) << print() << L" Successfully Uninitialized.";
    }

    std::wstring name() const override
    {
        return L"replay";
    }
};

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&                  params,
                                                    std::vector<spl::shared_ptr<core::video_channel>> channels)
{
    if (params.size() < 1 || !boost::iequals(params[0], L"REPLAY"))
        return core::frame_consumer::empty();

    std::wstring filename = L"REPLAY";

    short quality = REPLAY_JPEG_QUALITY;
    chroma_subsampling subsampling = REPLAY_JPEG_SUBSAMPLING;


    if (params.size() > 1)
    {
        filename = params[1];

        for (uint16_t i = 2; i<params.size(); i++)
        {
            if (boost::iequals(params[i], L"SUBSAMPLING"))
            {
                if (params[i + 1] == L"444")
                {
                    subsampling = Y444;
                    i++;
                }
                else if (params[i + 1] == L"422")
                {
                    subsampling = Y422;
                    i++;
                }
                else if (params[i + 1] == L"420")
                {
                    subsampling = Y420;
                    i++;
                }
                else if (params[i + 1] == L"411")
                {
                    subsampling = Y411;
                    i++;
                }
            }
            else if (boost::iequals(params[i], L"QUALITY"))
            {
                quality = boost::lexical_cast<short>(params[i + 1]);
                i++;
            }
        }
    }

    return spl::make_shared<replay_consumer>(filename, quality, subsampling, 2);
}

}}
