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

#include "file_operations.h"

#include <boost/shared_ptr.hpp>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

#include <common/env.h>
#ifndef CASPAR_2_1
#include <common/log/log.h>
#include <common/utility/string.h>
#include <common/concurrency/future_util.h>
#endif
#include <common/diagnostics/graph.h>
#include <core/consumer/frame_consumer.h>
#ifndef CASPAR_2_1
#include <core/mixer/read_frame.h>
#endif

#define VIDEO_OUTPUT_BUF_SIZE		4096
#define VIDEO_INPUT_BUF_SIZE		4096

#pragma warning(disable:4800)

namespace caspar { namespace replay {

	typedef struct tag_src_mgr {
		/// public fields
		struct jpeg_source_mgr pub;
		/// source stream
		mjpeg_file_handle infile;
		/// start of buffer
		JOCTET * buffer;
		/// have we gotten any data yet ?
		boolean start_of_file;
	} src_mgr;

	typedef struct tag_dest_mgr {
		/// public fields
		struct jpeg_destination_mgr pub;
		/// destination stream
		mjpeg_file_handle outfile;
		/// start of buffer
		JOCTET * buffer;
	} dest_mgr;

	typedef src_mgr*			src_ptr;
	typedef dest_mgr*			dest_ptr;

	typedef struct error_mgr * error_ptr;

#pragma warning(disable:4706)
	mjpeg_file_handle safe_fopen(const wchar_t* filename, uint32_t mode, uint32_t shareFlags)
	{
#ifdef REPLAY_IO_WINAPI
		//  | FILE_FLAG_NO_BUFFERING, FILE_FLAG_WRITE_THROUGH
		mjpeg_file_handle handle;
		if (handle = CreateFileW(filename, mode, shareFlags, NULL, (mode == GENERIC_WRITE ? CREATE_ALWAYS : OPEN_EXISTING), (mode == GENERIC_WRITE ? 0 : 0), NULL))
		{
			return handle;	
		}
		else
		{
			uint32_t error = GetLastError();
			printf("Error while opening file: %d (0x%x)", error, error);
			return NULL;
		}
#else
#ifndef CASPAR_2_1
		if (mode == GENERIC_WRITE)
			return fopen64(utf8util::UTF8FromUTF16(filename).c_str(), "wb");
		else
			return fopen64(utf8util::UTF8FromUTF16(filename).c_str(), "rb");
#else
                if (mode == GENERIC_WRITE)
                        return fopen64(u8(filename).c_str(), "wb");
                else
                        return fopen64(u8(filename).c_str(), "rb");
#endif
#endif
	}
#pragma warning(default:4706)

	void safe_fclose(mjpeg_file_handle file_handle)
	{
#ifdef REPLAY_IO_WINAPI
		CloseHandle(file_handle);
#else
		fclose(file_handle);
#endif;
	}

	void write_index_header(mjpeg_file_handle outfile_idx, const core::video_format_desc* format_desc, boost::posix_time::ptime start_timecode, int audio_channels)
	{
		mjpeg_file_header	header;
		header.magick[0] = 'O';	// Set the "magick" four bytes
		header.magick[1] = 'M';
		header.magick[2] = 'A';
		header.magick[3] = 'V';
		header.version = 2;	    // Set the version number to 2, for V.2 of the index file format (extended)
		header.width = format_desc->width;
		header.height = format_desc->height;
		header.fps = format_desc->fps;
		header.field_mode = format_desc->field_mode;
		header.begin_timecode = start_timecode;

		uint32_t written = 0;
#ifdef REPLAY_IO_WINAPI
		WriteFile(outfile_idx, &header, sizeof(mjpeg_file_header), (DWORD*)&written, NULL);
#else
		written = fwrite(&header, 1, sizeof(mjpeg_file_header), outfile_idx);
#endif

		mjpeg_file_header_ex	header_ex;
		header_ex.video_fourcc[0] = 'm'; // Set video fourcc
		header_ex.video_fourcc[1] = 'j';
		header_ex.video_fourcc[2] = 'p';
		header_ex.video_fourcc[3] = 'g';

		header_ex.audio_fourcc[0] = 'i';
		header_ex.audio_fourcc[1] = 'n';
		header_ex.audio_fourcc[2] = '3';
		header_ex.audio_fourcc[3] = '2';

		header_ex.audio_channels = audio_channels; // 2 for stereo

		written = 0;

#ifdef REPLAY_IO_WINAPI
		WriteFile(outfile_idx, &header_ex, sizeof(mjpeg_file_header_ex), (DWORD*)&written, NULL);
#else
		written = fwrite(&header_ex, 1, sizeof(mjpeg_file_header_ex), outfile_idx);
#endif
	}

	int read_index_header(mjpeg_file_handle infile_idx, mjpeg_file_header** header)
	{
		*header = new mjpeg_file_header();

		uint32_t read = 0;
		
#ifdef REPLAY_IO_WINAPI
		ReadFile(infile_idx, *header, sizeof(mjpeg_file_header), (DWORD*)&read, NULL);
#else
		read = fread(*header, 1, sizeof(mjpeg_file_header), infile_idx);
#endif
		if (read != sizeof(mjpeg_file_header))
		{
			delete *header;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	int read_index_header_ex(mjpeg_file_handle infile_idx, mjpeg_file_header_ex** header)
	{
		*header = new mjpeg_file_header_ex();

		uint32_t read = 0;

#ifdef REPLAY_IO_WINAPI
		ReadFile(infile_idx, *header, sizeof(mjpeg_file_header_ex), (DWORD*)&read, NULL);
#else
		read = fread(*header, 1, sizeof(mjpeg_file_header_ex), infile_idx);
#endif
		if (read != sizeof(mjpeg_file_header_ex))
		{
			delete *header;
			return 1;
		}
		else
		{
			return 0;
		}
	}

	void write_index(mjpeg_file_handle outfile_idx, long long offset)
	{
		/* fwrite(&offset, sizeof(long long), 1, outfile_idx.get());
		fflush(outfile_idx.get()); */
		uint32_t written = 0;
		//boost::timer perf_timer;
#ifdef REPLAY_IO_WINAPI
		WriteFile(outfile_idx, &offset, sizeof(long long), (DWORD*)&written, NULL);
#else
		written = fwrite(&offset, 1, sizeof(long long), outfile_idx);
#endif
		//if (write_time != NULL)
		//	*write_time = perf_timer.elapsed();
	}

	long long read_index(mjpeg_file_handle infile_idx)
	{
		long long offset = 0;
		uint32_t read = 0;
#ifdef REPLAY_IO_WINAPI
		ReadFile(infile_idx, &offset, sizeof(long long), (DWORD*)&read, NULL);
#else
		read = fread(&offset, 1, sizeof(long long), infile_idx);
#endif
		if (read)
		{
			return offset;
		}
		else
		{
			return -1;
		}
	}

	int seek_index(mjpeg_file_handle infile_idx, long long frame, uint32_t origin)
	{
#ifdef REPLAY_IO_WINAPI
		LARGE_INTEGER position;
#endif
		bool result;
		switch (origin)
		{
			case FILE_CURRENT:
				//return _fseeki64_nolock(infile_idx.get(), frame * sizeof(long long), SEEK_CUR);
#ifdef REPLAY_IO_WINAPI
				position.QuadPart = frame * sizeof(long long);
				result = SetFilePointerEx(infile_idx, position, NULL, FILE_CURRENT);
#else
				return fseek64(infile_idx, frame * sizeof(long long), SEEK_CUR);
#endif
				break;
			case FILE_BEGIN:
			default:
				//return _fseeki64_nolock(infile_idx.get(), frame * sizeof(long long) + sizeof(mjpeg_file_header), SEEK_SET);
#ifdef REPLAY_IO_WINAPI
				position.QuadPart = frame * sizeof(long long) + sizeof(mjpeg_file_header) + sizeof(mjpeg_file_header_ex);
				result = SetFilePointerEx(infile_idx, position, NULL, FILE_BEGIN);
#else
				return fseek64(infile_idx, frame * sizeof(long long) + sizeof(mjpeg_file_header) + sizeof(mjpeg_file_header_ex), SEEK_SET);
#endif
				break;
		}
		return (result ? 0 : 1);
	}

	int seek_frame(mjpeg_file_handle infile, long long offset, uint32_t origin = FILE_BEGIN /*deprecated*/)
	{
		//return _fseeki64_nolock(infile.get(), offset, origin);
#ifdef REPLAY_IO_WINAPI
		LARGE_INTEGER position;
		position.QuadPart = offset;
		bool result = SetFilePointerEx(infile, position, NULL, FILE_BEGIN);
		return (result ? 0 : 1);
#else
		return fseek64(infile, offset, SEEK_SET);
#endif
	}

	long long tell_index(mjpeg_file_handle infile_idx)
	{
		//return (_ftelli64_nolock(infile_idx.get()) - sizeof(mjpeg_file_header)) / sizeof(long long);
#ifdef REPLAY_IO_WINAPI
		LARGE_INTEGER zero;
		zero.QuadPart = 0;
		LARGE_INTEGER position;
		SetFilePointerEx(infile_idx, zero, &position, FILE_CURRENT);
		return (position.QuadPart - sizeof(mjpeg_file_header) - sizeof(mjpeg_file_header_ex)) / sizeof(long long);
#else
		return (ftell64(infile_idx) - sizeof(mjpeg_file_header) - sizeof(mjpeg_file_header_ex)) / sizeof(long long);
#endif
	}

	long long length_index(mjpeg_file_handle infile_idx)
	{
#ifdef REPLAY_IO_WINAPI
		LARGE_INTEGER size;
		GetFileSizeEx(infile_idx, &size);
		return (size.QuadPart - sizeof(mjpeg_file_header) - sizeof(mjpeg_file_header_ex)) / sizeof(long long);
#else
		long long pos = ftell64(infile_idx);
		fseek64(infile_idx, 0, SEEK_END);
		long long len = (ftell64(infile_idx) - sizeof(mjpeg_file_header) - sizeof(mjpeg_file_header_ex)) / sizeof(long long);
		fseek64(infile_idx, pos, SEEK_SET);
		return len;
#endif
	}

	long long tell_frame(mjpeg_file_handle infile)
	{
#ifdef REPLAY_IO_WINAPI
		LARGE_INTEGER zero;
		zero.QuadPart = 0;
		LARGE_INTEGER position;
		SetFilePointerEx(infile, zero, &position, FILE_CURRENT);
		return position.QuadPart;
#else
		return ftell64(infile);
#endif
	}

#pragma warning(disable: 4324)
	struct error_mgr {
		struct jpeg_error_mgr pub;	/* "public" fields */
		jmp_buf setjmp_buffer;	/* for return to caller */
	}; 
#pragma warning(default: 4324)

	typedef struct error_mgr * error_ptr;

	static void init_source (j_decompress_ptr cinfo)
	{
		src_ptr src = (src_ptr) cinfo->src;

		src->start_of_file = TRUE;
	}

	static boolean fill_input_buffer (j_decompress_ptr cinfo)
	{
		src_ptr src = (src_ptr) cinfo->src;

		//uint32_t nbytes = src->m_io->read_proc(src->buffer, 1, INPUT_BUF_SIZE, src->infile);

		uint32_t nbytes;
#ifdef REPLAY_IO_WINAPI
		ReadFile(src->infile, src->buffer, VIDEO_INPUT_BUF_SIZE, (DWORD*)&nbytes, NULL);
#else
		nbytes = fread(src->buffer, 1, VIDEO_INPUT_BUF_SIZE, src->infile);
#endif

		if (nbytes <= 0)
		{
			if (src->start_of_file)	/* Treat empty input file as fatal error */
				throw(cinfo, JERR_INPUT_EMPTY);

			WARNMS(cinfo, JWRN_JPEG_EOF);

			/* Insert a fake EOI marker */

			src->buffer[0] = (JOCTET) 0xFF;
			src->buffer[1] = (JOCTET) JPEG_EOI;

			nbytes = 2;
		}

		src->pub.next_input_byte = src->buffer;
		src->pub.bytes_in_buffer = nbytes;
		src->start_of_file = FALSE;

		return TRUE;
	}

	static void skip_input_data (j_decompress_ptr cinfo, long num_bytes)
	{
		src_ptr src = (src_ptr) cinfo->src;

		/* Just a dumb implementation for now.  Could use fseek() except
		 * it doesn't work on pipes.  Not clear that being smart is worth
		 * any trouble anyway --- large skips are infrequent.
		*/

		if (num_bytes > 0)
		{
			while (num_bytes > (long) src->pub.bytes_in_buffer)
			{
			  num_bytes -= (long) src->pub.bytes_in_buffer;

			  (void) fill_input_buffer(cinfo);

			  /* note we assume that fill_input_buffer will never return FALSE,
			   * so suspension need not be handled.
			   */
			}

			src->pub.next_input_byte += (uint32_t) num_bytes;
			src->pub.bytes_in_buffer -= (uint32_t) num_bytes;
		}
	}

	static void term_source (j_decompress_ptr cinfo)
	{
		// Nothing to actually do here
	}

	static void jpeg_windows_src (j_decompress_ptr cinfo, mjpeg_file_handle file)
	{
		src_ptr src;

		// allocate memory for the buffer. is released automatically in the end

		if (cinfo->src == NULL)
		{
			cinfo->src = (struct jpeg_source_mgr *) (*cinfo->mem->alloc_small)
				((j_common_ptr) cinfo, JPOOL_PERMANENT, sizeof(src_mgr));

			src = (src_ptr) cinfo->src;

			src->buffer = (JOCTET *) (*cinfo->mem->alloc_small)
				((j_common_ptr) cinfo, JPOOL_PERMANENT, VIDEO_INPUT_BUF_SIZE * sizeof(JOCTET));
		}

		// initialize the jpeg pointer struct with pointers to functions

		src = (src_ptr) cinfo->src;
		src->pub.init_source = init_source;
		src->pub.fill_input_buffer = fill_input_buffer;
		src->pub.skip_input_data = skip_input_data;
		src->pub.resync_to_restart = jpeg_resync_to_restart; // use default method 
		src->pub.term_source = term_source;
		src->infile = file;
		src->pub.bytes_in_buffer = 0;		// forces fill_input_buffer on first read 
		src->pub.next_input_byte = NULL;	// until buffer loaded 
	}

	static void init_destination (j_compress_ptr cinfo)
	{
		dest_ptr dest = (dest_ptr) cinfo->dest;

		dest->buffer = (JOCTET *)
		  (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
					  VIDEO_OUTPUT_BUF_SIZE * sizeof(JOCTET));

		dest->pub.next_output_byte = dest->buffer;
		dest->pub.free_in_buffer = VIDEO_OUTPUT_BUF_SIZE;
	}

	static void term_destination (j_compress_ptr cinfo)
	{
		dest_ptr dest = (dest_ptr) cinfo->dest;

		uint32_t datacount = VIDEO_OUTPUT_BUF_SIZE - (uint32_t)dest->pub.free_in_buffer;

		// write any data remaining in the buffer

		if (datacount > 0)
		{
			uint32_t written = 0;
			bool success = false;

#ifdef REPLAY_IO_WINAPI
			success = WriteFile(dest->outfile, dest->buffer, datacount, (DWORD*)&written, NULL);
#else
			written = fwrite(dest->buffer, 1, datacount, dest->outfile);
			if (written > 0)
				success = true;
#endif

			if (!success)
			  throw(cinfo, JERR_FILE_WRITE);
		}
	}

	static boolean empty_output_buffer (j_compress_ptr cinfo)
	{
		dest_ptr dest = (dest_ptr) cinfo->dest;

		uint32_t written = 0;
		bool success = false;
		
#ifdef REPLAY_IO_WINAPI
		success = WriteFile(dest->outfile, dest->buffer, VIDEO_OUTPUT_BUF_SIZE, (DWORD*)&written, NULL);
#else
		written = fwrite(dest->buffer, 1, VIDEO_OUTPUT_BUF_SIZE, dest->outfile);
		if (written > 0)
			success = true;
#endif

		if (!success)
			throw(cinfo, JERR_FILE_WRITE);

		dest->pub.next_output_byte = dest->buffer;
		dest->pub.free_in_buffer = VIDEO_OUTPUT_BUF_SIZE;

		return TRUE;
	}

	static void jpeg_windows_dest(j_compress_ptr cinfo, mjpeg_file_handle file)
	{
		dest_ptr dest;

		if (cinfo->dest == NULL)
		{
			cinfo->dest = (struct jpeg_destination_mgr *)(*cinfo->mem->alloc_small)
				((j_common_ptr) cinfo, JPOOL_PERMANENT, sizeof(dest_mgr));
		}

		dest = (dest_ptr) cinfo->dest;
		dest->pub.init_destination = init_destination;
		dest->pub.empty_output_buffer = empty_output_buffer;
		dest->pub.term_destination = term_destination;
		dest->outfile = file;
	}

	void error_exit(j_common_ptr cinfo)
	{
		/* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
		error_ptr myerr = (error_ptr) cinfo->err;
		/* Always display the message. */
		/* We could postpone this until after returning, if we chose. */
		(*cinfo->err->output_message) (cinfo);
		/* Return control to the setjmp point */
		longjmp(myerr->setjmp_buffer, 1);
	} 

	uint32_t read_frame(mjpeg_file_handle infile, uint32_t* width, uint32_t* height, uint8_t** image, uint32_t* audioSize, int32_t** audio)
	{
		uint32_t audioBufSize = 0;
		uint32_t read = 0;

#ifdef REPLAY_IO_WINAPI
		ReadFile(infile, &audioBufSize, sizeof(uint32_t), (DWORD*)&read, FALSE);
#else
		fread(&audioBufSize, 1, sizeof(uint32_t), infile);
#endif

		if (audioBufSize > 0)
		{
			read = 0;
			(*audio) = new int32_t[audioBufSize/4];
#ifdef REPLAY_IO_WINAPI
			ReadFile(infile, *audio, (DWORD)audioBufSize, (DWORD*)&read, FALSE);
#else
			read = fread(*audio, 1, audioBufSize, infile);
#endif
		}

		(*audioSize) = audioBufSize;

		struct jpeg_decompress_struct cinfo;

		struct error_mgr jerr;

		JSAMPROW row_pointer[1];
		int row_stride;

		cinfo.err = jpeg_std_error(&jerr.pub);
		jerr.pub.error_exit = error_exit;
		/* Establish the setjmp return context for my_error_exit to use. */
#pragma warning(disable: 4611)
		if (setjmp(jerr.setjmp_buffer)) {
			/* If we get here, the JPEG code has signaled an error.
			* We need to clean up the JPEG object, close the input file, and return.
			*/
			jpeg_destroy_decompress(&cinfo);
			return 0;
		}
#pragma warning(default: 4611)
		jpeg_create_decompress(&cinfo);

		//jpeg_stdio_src(&cinfo, infile.get());
		jpeg_windows_src(&cinfo, infile);

		(void) jpeg_read_header(&cinfo, TRUE); // We ignore the return value - all errors will result in exiting as per setjmp error handler

		(void) jpeg_start_decompress(&cinfo);

		row_stride = cinfo.output_width * 3;

		(*width) = cinfo.output_width;
		(*height) = cinfo.output_height;
		(*image) = new uint8_t[(*width) * (*height) * 3];

		while (cinfo.output_scanline < cinfo.output_height)
		{
			row_pointer[0] = (JSAMPROW)((*image) + (cinfo.output_scanline * row_stride));
			(void) jpeg_read_scanlines(&cinfo, row_pointer, 1);
		} 

		(void) jpeg_finish_decompress(&cinfo);

		jpeg_destroy_decompress(&cinfo);

		return ((*width) * (*height) * 3);
	}

	#pragma warning(disable:4267)
	long long write_frame(mjpeg_file_handle outfile, uint32_t width, uint32_t height, const uint8_t* image, short quality, mjpeg_process_mode mode, chroma_subsampling subsampling, const int32_t* audio_data, uint32_t audio_data_length)
	{
		long long start_position = tell_frame(outfile);

#ifdef REPLAY_IO_WINAPI
		uint32_t written = 0;
		WriteFile(outfile, &audio_data_length, sizeof(uint32_t), (DWORD*)&written, NULL);
		WriteFile(outfile, audio_data, audio_data_length, (DWORD*)&written, NULL);
#else
		fwrite(&audio_data_length, 1, sizeof(uint32_t), outfile);
		fwrite(audio_data, 1, audio_data_length, outfile);
#endif

		// JPEG Compression Parameters
		struct jpeg_compress_struct cinfo;
		// JPEG error info
		struct jpeg_error_mgr jerr;

		JSAMPROW row_pointer[1];
		uint32_t row_stride;

		cinfo.err = jpeg_std_error(&jerr);
		jpeg_create_compress(&cinfo);

		//jpeg_stdio_dest(&cinfo, outfile.get());
		jpeg_windows_dest(&cinfo, outfile);

		cinfo.image_width = width;
		cinfo.image_height = height;
		cinfo.input_components = 4;
		cinfo.in_color_space = JCS_EXT_BGRX;

		cinfo.max_v_samp_factor = 1;
		cinfo.max_h_samp_factor = 1;
		cinfo.jpeg_color_space = JCS_YCbCr;

		jpeg_set_defaults(&cinfo);

		jpeg_set_quality(&cinfo, quality, TRUE);

		if (subsampling == Y444)
		{
			cinfo.comp_info[0].h_samp_factor = 1;
			cinfo.comp_info[0].v_samp_factor = 1;
			cinfo.comp_info[1].h_samp_factor = 1;
			cinfo.comp_info[1].v_samp_factor = 1;
			cinfo.comp_info[2].h_samp_factor = 1;
			cinfo.comp_info[2].v_samp_factor = 1;
		}
		else if (subsampling == Y422)
		{
			cinfo.comp_info[0].h_samp_factor = 2;
			cinfo.comp_info[0].v_samp_factor = 1;
			cinfo.comp_info[1].h_samp_factor = 1;
			cinfo.comp_info[1].v_samp_factor = 1;
			cinfo.comp_info[2].h_samp_factor = 1;
			cinfo.comp_info[2].v_samp_factor = 1;
		}
		else if (subsampling == Y420)
		{
			cinfo.comp_info[0].h_samp_factor = 2;
			cinfo.comp_info[0].v_samp_factor = 2;
			cinfo.comp_info[1].h_samp_factor = 1;
			cinfo.comp_info[1].v_samp_factor = 1;
			cinfo.comp_info[2].h_samp_factor = 1;
			cinfo.comp_info[2].v_samp_factor = 1;
		}
		else if (subsampling == Y411)
		{
			cinfo.comp_info[0].h_samp_factor = 4;
			cinfo.comp_info[0].v_samp_factor = 1;
			cinfo.comp_info[1].h_samp_factor = 1;
			cinfo.comp_info[1].v_samp_factor = 1;
			cinfo.comp_info[2].h_samp_factor = 1;
			cinfo.comp_info[2].v_samp_factor = 1;
		}

		jpeg_start_compress(&cinfo, TRUE);

		// JSAMPLEs per row in image_buffer
		row_stride = width * 4;

		if (mode == PROGRESSIVE) {
			while (cinfo.next_scanline < cinfo.image_height)
			{
				row_pointer[0] = (JSAMPROW)(image + (cinfo.next_scanline * row_stride));
				(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
			}
		}
		else if (mode == UPPER)
		{
			while (cinfo.next_scanline < cinfo.image_height)
			{
				row_pointer[0] = (JSAMPROW)(image + ((cinfo.next_scanline * 2) * row_stride));
				(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
			}
		}
		else if (mode == LOWER)
		{
			while (cinfo.next_scanline < cinfo.image_height)
			{
				row_pointer[0] = (JSAMPROW)(image + ((cinfo.next_scanline * 2 + 1) * row_stride));
				(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
			}
		}
		
		jpeg_finish_compress(&cinfo);
				
		jpeg_destroy_compress(&cinfo); 

		/* if (compress_time != NULL)
		{
			*compress_time = perf_timer.elapsed();
		} */

		return start_position;
	}
#pragma warning(default:4267)

}}