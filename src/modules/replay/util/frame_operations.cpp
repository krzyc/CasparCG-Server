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

#include "frame_operations.h"

#include <memory>
#include <tmmintrin.h>
#include <tbb/parallel_for.h>

namespace caspar { namespace replay {

void interlace_fields(const mmx_uint8_t* src1, const mmx_uint8_t* src2, mmx_uint8_t* dst, uint32_t width, uint32_t height, uint32_t stride)
{
    uint32_t full_row = width * stride;
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, height/2), [=](const tbb::blocked_range<uint32_t>& r)
    {
        for (auto i = r.begin(); i != r.end(); ++i)
        {
            memcpy((dst + i * 2 * full_row), (src1 + i * full_row), full_row);
            memcpy((dst + (i * 2 + 1) * full_row), (src2 + i * full_row), full_row);
        }
    });
}

void interlace_frames(const mmx_uint8_t* src1, const mmx_uint8_t* src2, mmx_uint8_t* dst, uint32_t width, uint32_t height, uint32_t stride)
{
    uint32_t full_row = width * stride;
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, height/2), [=](const tbb::blocked_range<uint32_t>& r)
    {
        for (auto i = r.begin(); i != r.end(); ++i)
        {
            memcpy((dst + i * 2 * full_row), (src1 + i * 2 * full_row), full_row);
            memcpy((dst + (i * 2 + 1) * full_row), (src2 + (i * 2 + 1) * full_row), full_row);
        }
    });
}

void field_double(const mmx_uint8_t* src, mmx_uint8_t* dst, uint32_t width, uint32_t height, uint32_t stride)
{
    uint32_t full_row = width * stride;
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, height/2 - 1), [=](const tbb::blocked_range<uint32_t>& r)
    {
        for (auto i = r.begin(); i != r.end(); ++i)
        {
            for (uint32_t j = 0; j < full_row; ++j)
            {
                dst[i * 2 * full_row + j] = src[i * full_row + j];
                dst[(i * 2 + 1) * full_row + j] = (src[i * full_row + j] >> 1) + (src[(i + 1) * full_row + j] >> 1);
            }
        }
    });
}

#pragma warning(disable:4309 4244)
// max level is 63
// level = 63 means 100% src1, level = 0 means 100% src2
void blend_images(const mmx_uint8_t* src1, mmx_uint8_t* src2, mmx_uint8_t* dst, uint32_t width, uint32_t height, uint32_t stride, uint8_t level)
{
    uint32_t full_size = width * height * stride;
    uint16_t level_16 = (uint16_t)level;
#ifndef OPTIMIZE_BLEND_IMAGES
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, full_size), [=](const tbb::blocked_range<uint32_t>& r)
    {
        for (auto i = r.begin(); i != r.end(); i++)
        {
            dst[i] = (uint8_t)((((int)src1[i] * level_16) >> 6) + (((int)src2[i] * (64 - level_16)) >> 6));
        }
    });
#else
    const __m64* in1_vec = (__m64*)src1;
    const __m64* in2_vec = (__m64*)src2;
    __m64* out_vec = (__m64*)dst;
    __m128i mul = _mm_set_epi16(level_16, level_16, level_16, level_16, level_16, level_16, level_16, level_16);
    __m128i in1, in2;
    __m128i mask = _mm_set_epi8(0x80, 7, 0x80, 6, 0x80, 5, 0x80, 4, 0x80, 3, 0x80, 2, 0x80, 1, 0x80, 0);
    __m128i umask = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    full_size /= 24;
    while (full_size-- > 0)
    {
        /*             01 23 45 67 89 AB CD EF
            * in_vec[0]   Ra Ga Ba Rb Gb Bb Rc Gc
            * in_vec[1]   Bc Rd Gd Bd Re Ge Be Rf
            * in_vec[2]   Gg Bg Rh Gh Bh Ri Gi Bi
            */

        in1 = _mm_loadl_epi64((__m128i*)&in1_vec[0]);
        in2 = _mm_loadl_epi64((__m128i*)&in2_vec[0]);
        in1 = _mm_shuffle_epi8(in1, mask);
        in2 = _mm_shuffle_epi8(in2, mask);
        in1 = _mm_sub_epi16(in1, in2);
        in1 = _mm_mullo_epi16 (in1, mul);
        in1 = _mm_srli_epi16(in1, 6);
        in1 = _mm_add_epi16(in1, in2);
        in1 = _mm_shuffle_epi8(in1, umask);
        _mm_storel_epi64((__m128i*)&out_vec[0], in1);

        in1 = _mm_loadl_epi64((__m128i*)&in1_vec[1]);
        in2 = _mm_loadl_epi64((__m128i*)&in2_vec[1]);
        in1 = _mm_shuffle_epi8(in1, mask);
        in2 = _mm_shuffle_epi8(in2, mask);
        in1 = _mm_sub_epi16(in1, in2);
        in1 = _mm_mullo_epi16 (in1, mul);
        in1 = _mm_srli_epi16(in1, 6);
        in1 = _mm_add_epi16(in1, in2);
        in1 = _mm_shuffle_epi8(in1, umask);
        _mm_storel_epi64((__m128i*)&out_vec[1], in1);

        in1 = _mm_loadl_epi64((__m128i*)&in1_vec[2]);
        in2 = _mm_loadl_epi64((__m128i*)&in2_vec[2]);
        in1 = _mm_shuffle_epi8(in1, mask);
        in2 = _mm_shuffle_epi8(in2, mask);
        in1 = _mm_sub_epi16(in1, in2);
        in1 = _mm_mullo_epi16 (in1, mul);
        in1 = _mm_srli_epi16(in1, 6);
        in1 = _mm_add_epi16(in1, in2);
        in1 = _mm_shuffle_epi8(in1, umask);
        _mm_storel_epi64((__m128i*)&out_vec[2], in1);

        in1_vec += 3;
        in2_vec += 3;
        out_vec += 3;
    }
#endif
}

void black_frame(mmx_uint8_t* dst, uint32_t width, uint32_t height, uint32_t stride)
{
    uint32_t full_size = width * height;
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, full_size), [=](const tbb::blocked_range<uint32_t>& r)
    {
        for (auto i = r.begin(); i != r.end(); i++)
        {
            dst[i*stride] = 0;
            dst[i*stride+1] = 0;
            dst[i*stride+2] = 0;
            if (stride == 4)
                dst[i*stride+3] = 255;
        }
    });
}
#pragma warning(default:4309)

}}
