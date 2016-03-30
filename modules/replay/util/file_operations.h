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

#pragma once

#include <stdint.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <core/video_format.h>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
// comment below line to use STDIO file operations on Windows
#define REPLAY_IO_WINAPI
#endif

#ifndef REPLAY_IO_WINAPI

#ifndef CASPAR_2_1
#include "common/utility/utf8conv.h"
#else
#include "common/utf.h"
#endif

#define _FILE_OFFSET_BITS  64

#ifdef _WIN32
#ifndef fopen64
#define fopen64 fopen
#endif
#ifndef fseek64
#define fseek64 _fseeki64
#endif
#ifndef ftell64
#define ftell64 _ftelli64
#endif
#endif

#ifdef __x86_64
#ifndef fseek64
#define fseek64 fseek
#endif
#ifndef ftell64
#define ftell64 ftell
#endif
#endif

#ifndef FILE_CURRENT
#define FILE_CURRENT SEEK_CUR
#endif
#ifndef FILE_BEGIN
#define FILE_BEGIN SEEK_SET
#endif
#ifndef GENERIC_READ
#define GENERIC_READ 0x80000000
#endif
#ifndef GENERIC_WRITE
#define GENERIC_WRITE 0x40000000
#endif
#ifndef FILE_SHARE_READ
#define FILE_SHARE_READ 0x00000001
#endif
#ifndef FILE_SHARE_WRITE
#define FILE_SHARE_WRITE 0x00000002
#endif
#endif

#ifdef REPLAY_IO_WINAPI
typedef HANDLE						mjpeg_file_handle;
#else
typedef FILE*						mjpeg_file_handle;
#endif

namespace caspar { namespace replay {

	struct mjpeg_file_header {
		char							magick[4]; // = 'OMAV'
		uint8_t							version; // = 1 for version 1, or 2 for version 2
		uint32_t						width;
		uint32_t						height;
		double							fps;
#ifndef CASPAR_2_1
		caspar::core::field_mode::type	field_mode;
#else
		caspar::core::field_mode		field_mode;
#endif
		boost::posix_time::ptime		begin_timecode;
	};

	// Extended header used in version 2
	struct mjpeg_file_header_ex {
		char							video_fourcc[4]; // = 'mjpg'
		char							audio_fourcc[4]; // = 'in32'

		int								audio_channels;
	};

	enum mjpeg_process_mode
	{
		PROGRESSIVE,
		UPPER,
		LOWER
	};

	enum chroma_subsampling
	{
		Y444,
		Y422,
		Y420,
		Y411
	};

	mjpeg_file_handle safe_fopen(const wchar_t* filename, uint32_t mode, uint32_t shareFlags);
	void safe_fclose(mjpeg_file_handle file_handle);
	void write_index_header(mjpeg_file_handle outfile_idx, const core::video_format_desc* format_desc, boost::posix_time::ptime start_timecode, int audio_channels);
	void write_index(mjpeg_file_handle outfile_idx, long long offset);
	long long write_frame(mjpeg_file_handle outfile, uint32_t width, uint32_t height, const uint8_t* image, short quality, mjpeg_process_mode mode, chroma_subsampling subsampling, const int32_t* audio_data, uint32_t audio_data_length);
	long long read_index(mjpeg_file_handle infile_idx);
	long long tell_index(mjpeg_file_handle infile_idx);
	long long length_index(mjpeg_file_handle infile_idx);
	int seek_index(mjpeg_file_handle infile_idx, long long frame, uint32_t origin);
	long long tell_frame(mjpeg_file_handle infile);
	int read_index_header(mjpeg_file_handle infile_idx, mjpeg_file_header** header);
	int read_index_header_ex(mjpeg_file_handle infile_idx, mjpeg_file_header_ex** header);
	uint32_t read_frame(mjpeg_file_handle infile, uint32_t* width, uint32_t* height, uint8_t** image, uint32_t* audioSize, int32_t** audio);
	int seek_frame(mjpeg_file_handle infile, long long offset, uint32_t origin);
}}
