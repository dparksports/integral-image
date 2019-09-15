void integral_sum(const uint8_t * src, size_t src_stride, size_t width, size_t height, uint32_t * sum, size_t sum_stride)
{
    __m512i MASK = _mm_setr_epi64(
        0x00000000000000FF, 0x000000000000FFFF, 0x0000000000FFFFFF, 0x00000000FFFFFFFF
        0xFFFFFFFFFFFFFFFF, 0x00FFFFFFFFFFFFFF, 0x0000FFFFFFFFFFFF, 0x000000FFFFFFFFFF);
    __m512i K_15 = _mm512_set1_epi32(15);
    __m512i ZERO = _mm512_set1_epi32(0);

    memset(sum, 0, (width + 1)*sizeof(uint32_t));
    sum += sum_stride + 1;
    size_t aligned_width = width/8*8;

    for(size_t row = 0; row < height; row++)
    {
        sum[-1] = 0;
        size_t col = 0;
        __m512i row_sums = ZERO;
        for(; col < aligned_width; col += 8)
        {
            __m512i _src = _mm512_and_si512(_mm512_set1_epi32(*(uint32_t*)(src + col)), MASK);
            row_sums = _mm512_add_epi512(row_sums, _mm512_sad_epu8(_src, ZERO));
            __m256i curr_row_sums = _mm512_cvtepi64_epi32(row_sums);
            __m256i prev_row_sums = _mm256_loadu_si256((__m256i*)(sum + col - sum_stride));
            _mm_storeu_si128((__m128i*)(sum + col), _mm_add_epi32(curr_row_sums, prev_row_sums));
            row_sums = _mm512_permutexvar_epi64(row_sums, K_15);
        }
        uint32_t row_sum = sum[col - 1] - sum[col - sum_stride - 1];
        for (; col < width; col++)
        {
            row_sum += src[col];
            sum[col] = row_sum + sum[col - sum_stride];
        }
        src += src_stride;
        sum += sum_stride;
    }
}
