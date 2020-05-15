#include <cstdint>
#include <cstring>
#include <avisynth.h>

#define DECROSS_X86
#if defined (DECROSS_X86)
#include <emmintrin.h>
#endif

#ifdef _WIN32
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

static FORCE_INLINE bool Diff(const uint8_t* pDiff0, const uint8_t* pDiff1, const int nPos, int& nMiniDiff) {
#if defined (DECROSS_X86)
	__m128i mDiff0 = _mm_loadl_epi64((const __m128i*) & pDiff0[nPos]);
	__m128i mDiff1 = _mm_loadl_epi64((const __m128i*)pDiff1);

	__m128i mDiff = _mm_sad_epu8(mDiff0, mDiff1);

	int nDiff = _mm_cvtsi128_si32(mDiff);
	if (nDiff < nMiniDiff) {
		nMiniDiff = nDiff;
		return true;
	}
	return false;
#else
	int nDiff = 0;

	for (int i = 0; i < 8; i++)
		nDiff += std::abs(pDiff0[i + nPos] - pDiff1[i]);

	if (nDiff < nMiniDiff) {
		nMiniDiff = nDiff;
		return true;
	}
	return false;
#endif
}

static FORCE_INLINE void EdgeCheck(const uint8_t* pSrc, uint8_t* pEdgeBuffer, const int nRowSizeU, const int nYThreshold, const int nMargin) {
#if defined (DECROSS_X86)
	__m128i mYThreshold = _mm_set1_epi8(nYThreshold - 128);
	__m128i bytes_128 = _mm_set1_epi8(static_cast<char>(128));

	for (int nX = 4; nX < nRowSizeU - 4; nX += 4) {
		__m128i mLeft = _mm_loadl_epi64((const __m128i*) & pSrc[nX * 2 - 1]);
		__m128i mCenter = _mm_loadl_epi64((const __m128i*) & pSrc[nX * 2]);
		__m128i mRight = _mm_loadl_epi64((const __m128i*) & pSrc[nX * 2 + 1]);

		__m128i mLeft_128 = _mm_sub_epi8(mLeft, bytes_128);
		__m128i mCenter_128 = _mm_sub_epi8(mCenter, bytes_128);
		__m128i mRight_128 = _mm_sub_epi8(mRight, bytes_128);

		__m128i abs_diff_left_right = _mm_or_si128(_mm_subs_epu8(mLeft, mRight),
			_mm_subs_epu8(mRight, mLeft));
		abs_diff_left_right = _mm_sub_epi8(abs_diff_left_right, bytes_128);

		__m128i mEdge = _mm_and_si128(_mm_cmpgt_epi8(abs_diff_left_right, mYThreshold),
			_mm_or_si128(_mm_and_si128(_mm_cmpgt_epi8(mCenter_128, mLeft_128),
				_mm_cmpgt_epi8(mRight_128, mCenter_128)),
				_mm_and_si128(_mm_cmpgt_epi8(mLeft_128, mCenter_128),
					_mm_cmpgt_epi8(mCenter_128, mRight_128))));

		mEdge = _mm_packs_epi16(mEdge, mEdge);

		for (int i = -nMargin; i <= nMargin; i++) {
			*(int*)& pEdgeBuffer[nX + i] = _mm_cvtsi128_si32(_mm_or_si128(_mm_cvtsi32_si128(*(const int*)& pEdgeBuffer[nX + i]),
				mEdge));
		}
	}
#else
	for (int x = 4; x < nRowSizeU - 4; x++) {
		int left = pSrc[x * 2 - 1];
		int center = pSrc[x * 2];
		int right = pSrc[x * 2 + 1];

		bool edge =
			std::abs(left - right) > nYThreshold &&
			((center > left && right > center) || (left > center && center > right));

		for (int i = -nMargin; i <= nMargin; i++)
			pEdgeBuffer[x + i] = pEdgeBuffer[x + i] || edge;
	}
#endif
}

static FORCE_INLINE void AverageChroma(const uint8_t* pSrcU, const uint8_t* pSrcV, const uint8_t* pSrcUMini, const uint8_t* pSrcVMini, uint8_t* pDestU, uint8_t* pDestV, const uint8_t* pEdgeBuffer, int nX) {
#if defined (DECROSS_X86)
	__m128i mSrcU = _mm_cvtsi32_si128(*(const int*)& pSrcU[nX]);
	__m128i mSrcV = _mm_cvtsi32_si128(*(const int*)& pSrcV[nX]);

	__m128i mSrcUMini = _mm_cvtsi32_si128(*(const int*)& pSrcUMini[nX]);
	__m128i mSrcVMini = _mm_cvtsi32_si128(*(const int*)& pSrcVMini[nX]);

	__m128i mEdge = _mm_cvtsi32_si128(*(const int*)& pEdgeBuffer[nX]);

	__m128i mBlendColorU = _mm_avg_epu8(mSrcU, mSrcUMini);
	__m128i mBlendColorV = _mm_avg_epu8(mSrcV, mSrcVMini);

	__m128i mask = _mm_cmpeq_epi8(mEdge, _mm_setzero_si128());

	__m128i mDestU = _mm_or_si128(_mm_and_si128(mask, mSrcU),
		_mm_andnot_si128(mask, mBlendColorU));
	__m128i mDestV = _mm_or_si128(_mm_and_si128(mask, mSrcV),
		_mm_andnot_si128(mask, mBlendColorV));

	*(int*)& pDestU[nX] = _mm_cvtsi128_si32(mDestU);
	*(int*)& pDestV[nX] = _mm_cvtsi128_si32(mDestV);
#else
	for (int i = 0; i < 4; i++) {
		if (pEdgeBuffer[nX + i] == 0) {
			pDestU[nX + i] = pSrcU[nX + i];
			pDestV[nX + i] = pSrcV[nX + i];
		}
		else {
			pDestU[nX + i] = (pSrcU[nX + i] + pSrcUMini[nX + i] + 1) >> 1;
			pDestV[nX + i] = (pSrcV[nX + i] + pSrcVMini[nX + i] + 1) >> 1;
		}
	}
#endif
}

class DeCross : public GenericVideoFilter {
	int nYThreshold;
	int nNoiseThreshold;
	int nMargin;
	bool bDebug;

public:
	DeCross(PClip _child, int thresholdy, int noise, int margin, bool debug)
		: GenericVideoFilter(_child), nYThreshold(thresholdy), nNoiseThreshold(noise), nMargin(margin), bDebug(debug)
	{}

	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
	{
		PVideoFrame srcFrame = child->GetFrame(n, env);

		if (n == 0 || n >= vi.num_frames - 1)
			return srcFrame;

		PVideoFrame srcP = child->GetFrame(n - 1, env);
		PVideoFrame srcF = child->GetFrame(n + 1, env);
		PVideoFrame destFrame = srcFrame;
		env->MakeWritable(&destFrame);

		const int nHeightU = srcFrame->GetHeight(PLANAR_U);
		const int nRowSizeU = srcFrame->GetRowSize(PLANAR_U);
		const int nSrcPitch = srcFrame->GetPitch();
		const int nSrcPitch2 = nSrcPitch * 2;
		const int nSrcPitchU = srcFrame->GetPitch(PLANAR_U);
		const int nDestPitchU = destFrame->GetPitch(PLANAR_U);

		int subSamplingH;

		if (vi.Is420())
		{
			subSamplingH = 1;
		}
		else
		{
			subSamplingH = 0;
		}

		const uint8_t* pSrc = srcFrame->GetReadPtr() + nSrcPitch2;
		const uint8_t* pSrcP = srcP->GetReadPtr() + nSrcPitch2;
		const uint8_t* pSrcF = srcF->GetReadPtr() + nSrcPitch2;

		const uint8_t* pSrcTT = pSrc - nSrcPitch2;
		const uint8_t* pSrcBB = pSrc + nSrcPitch2;
		const uint8_t* pSrcPTT = pSrcP - nSrcPitch2;
		const uint8_t* pSrcPBB = pSrcP + nSrcPitch2;
		const uint8_t* pSrcFTT = pSrcF - nSrcPitch2;
		const uint8_t* pSrcFBB = pSrcF + nSrcPitch2;

		const uint8_t* pSrcT = pSrc - nSrcPitch;
		const uint8_t* pSrcB = pSrc + nSrcPitch;
		const uint8_t* pSrcPT = pSrcP - nSrcPitch;
		const uint8_t* pSrcPB = pSrcP + nSrcPitch;
		const uint8_t* pSrcFT = pSrcF - nSrcPitch;
		const uint8_t* pSrcFB = pSrcF + nSrcPitch;

		const uint8_t* pSrcU = srcFrame->GetReadPtr(PLANAR_U) + nSrcPitchU;
		const uint8_t* pSrcUP = srcP->GetReadPtr(PLANAR_U) + nSrcPitchU;
		const uint8_t* pSrcUF = srcF->GetReadPtr(PLANAR_U) + nSrcPitchU;
		const uint8_t* pSrcV = srcFrame->GetReadPtr(PLANAR_V) + nSrcPitchU;
		const uint8_t* pSrcVP = srcP->GetReadPtr(PLANAR_V) + nSrcPitchU;
		const uint8_t* pSrcVF = srcF->GetReadPtr(PLANAR_V) + nSrcPitchU;

		const uint8_t* pSrcUTT = pSrcU - nSrcPitchU;
		const uint8_t* pSrcUBB = pSrcU + nSrcPitchU;
		const uint8_t* pSrcUPTT = pSrcUP - nSrcPitchU;
		const uint8_t* pSrcUPBB = pSrcUP + nSrcPitchU;
		const uint8_t* pSrcUFTT = pSrcUF - nSrcPitchU;
		const uint8_t* pSrcUFBB = pSrcUF + nSrcPitchU;
		const uint8_t* pSrcVTT = pSrcV - nSrcPitchU;
		const uint8_t* pSrcVBB = pSrcV + nSrcPitchU;
		const uint8_t* pSrcVPTT = pSrcVP - nSrcPitchU;
		const uint8_t* pSrcVPBB = pSrcVP + nSrcPitchU;
		const uint8_t* pSrcVFTT = pSrcVF - nSrcPitchU;
		const uint8_t* pSrcVFBB = pSrcVF + nSrcPitchU;

		const uint8_t* pSrcUMini;
		const uint8_t* pSrcVMini;

		uint8_t* pDestU = destFrame->GetWritePtr(PLANAR_U) + nDestPitchU;
		uint8_t* pDestV = destFrame->GetWritePtr(PLANAR_V) + nDestPitchU;

		uint8_t* pEdgeBuffer = (uint8_t*)malloc(nRowSizeU);

		int skip = 1 << subSamplingH;

		for (int nY = nHeightU - skip; nY > skip; nY--) {
			memset(pEdgeBuffer, 0, nRowSizeU);

			EdgeCheck(pSrc, pEdgeBuffer, nRowSizeU, nYThreshold, nMargin);

			if (bDebug) {
				for (int nX = 4; nX < nRowSizeU - 4; nX++) {
					if (pEdgeBuffer[nX] != 0) {
						pDestU[nX] = 128;
						pDestV[nX] = 255;
					}
				}
			}
			else {
				int nX2 = 0;
				for (int nX = 4; nX < nRowSizeU - 4; nX += 4) {
					nX2 += 4 * 2;
					if (*(int*)& pEdgeBuffer[nX] != 0) {
						int nMiniDiff = nNoiseThreshold;
						pSrcUMini = pSrcU;
						pSrcVMini = pSrcV;

						if (nY % 2 == 1) {
							if (Diff(pSrcPTT + nX2, pSrcT + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUPTT - 3; pSrcVMini = pSrcVPTT - 3; }
							if (Diff(pSrcPTT + nX2, pSrcT + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUPTT - 1; pSrcVMini = pSrcVPTT - 1; }
							if (Diff(pSrcPT + nX2, pSrcT + nX2, -4, nMiniDiff)) { pSrcUMini = pSrcUP - 2; pSrcVMini = pSrcVP - 2; }
							if (Diff(pSrcPBB + nX2, pSrcT + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUPBB - 3; pSrcVMini = pSrcVPBB - 3; }
							if (Diff(pSrcPBB + nX2, pSrcT + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUPBB - 1; pSrcVMini = pSrcVPBB - 1; }

							if (Diff(pSrcTT + nX2, pSrcT + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUTT - 3; pSrcVMini = pSrcVTT - 3; }
							if (Diff(pSrcTT + nX2, pSrcT + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUTT - 1; pSrcVMini = pSrcVTT - 1; }
							if (Diff(pSrcT + nX2, pSrcT + nX2, -4, nMiniDiff)) { pSrcUMini = pSrcU - 2; pSrcVMini = pSrcV - 2; }
							if (Diff(pSrcBB + nX2, pSrcT + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUBB - 3; pSrcVMini = pSrcVBB - 3; }
							if (Diff(pSrcBB + nX2, pSrcT + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUBB - 1; pSrcVMini = pSrcVBB - 1; }

							if (Diff(pSrcFTT + nX2, pSrcT + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUFTT - 3; pSrcVMini = pSrcVFTT - 3; }
							if (Diff(pSrcFTT + nX2, pSrcT + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUFTT - 1; pSrcVMini = pSrcVFTT - 1; }
							if (Diff(pSrcFT + nX2, pSrcT + nX2, -4, nMiniDiff)) { pSrcUMini = pSrcUF - 2; pSrcVMini = pSrcVF - 2; }
							if (Diff(pSrcFBB + nX2, pSrcT + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUFBB - 3; pSrcVMini = pSrcVFBB - 3; }
							if (Diff(pSrcFBB + nX2, pSrcT + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUFBB - 1; pSrcVMini = pSrcVFBB - 1; }

							if (Diff(pSrcPT + nX2, pSrcT + nX2, -0, nMiniDiff)) { pSrcUMini = pSrcUP - 0; pSrcVMini = pSrcVP - 0; }
							if (Diff(pSrcFT + nX2, pSrcT + nX2, -0, nMiniDiff)) { pSrcUMini = pSrcUF - 0; pSrcVMini = pSrcVF - 0; }
							if (Diff(pSrcPB + nX2, pSrcB + nX2, -0, nMiniDiff)) { pSrcUMini = pSrcUP - 0; pSrcVMini = pSrcVP - 0; }
							if (Diff(pSrcFB + nX2, pSrcB + nX2, -0, nMiniDiff)) { pSrcUMini = pSrcUF - 0; pSrcVMini = pSrcVF - 0; }

							if (Diff(pSrcPTT + nX2, pSrcT + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUPTT + 3; pSrcVMini = pSrcVPTT + 3; }
							if (Diff(pSrcPTT + nX2, pSrcT + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUPTT + 1; pSrcVMini = pSrcVPTT + 1; }
							if (Diff(pSrcPT + nX2, pSrcT + nX2, +4, nMiniDiff)) { pSrcUMini = pSrcUP + 2; pSrcVMini = pSrcVP + 2; }
							if (Diff(pSrcPBB + nX2, pSrcT + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUPBB + 3; pSrcVMini = pSrcVPBB + 3; }
							if (Diff(pSrcPBB + nX2, pSrcT + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUPBB + 1; pSrcVMini = pSrcVPBB + 1; }

							if (Diff(pSrcTT + nX2, pSrcT + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUTT + 3; pSrcVMini = pSrcVTT + 3; }
							if (Diff(pSrcTT + nX2, pSrcT + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUTT + 1; pSrcVMini = pSrcVTT + 1; }
							if (Diff(pSrcT + nX2, pSrcT + nX2, +4, nMiniDiff)) { pSrcUMini = pSrcU + 2; pSrcVMini = pSrcV + 2; }
							if (Diff(pSrcBB + nX2, pSrcT + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUBB + 3; pSrcVMini = pSrcVBB + 3; }
							if (Diff(pSrcBB + nX2, pSrcT + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUBB + 1; pSrcVMini = pSrcVBB + 1; }

							if (Diff(pSrcFTT + nX2, pSrcT + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUFTT + 3; pSrcVMini = pSrcVFTT + 3; }
							if (Diff(pSrcFTT + nX2, pSrcT + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUFTT + 1; pSrcVMini = pSrcVFTT + 1; }
							if (Diff(pSrcFT + nX2, pSrcT + nX2, +4, nMiniDiff)) { pSrcUMini = pSrcUF + 2; pSrcVMini = pSrcVF + 2; }
							if (Diff(pSrcFBB + nX2, pSrcT + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUFBB + 3; pSrcVMini = pSrcVFBB + 3; }
							if (Diff(pSrcFBB + nX2, pSrcT + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUFBB + 1; pSrcVMini = pSrcVFBB + 1; }
						}
						else {
							if (Diff(pSrcPT + nX2, pSrc + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUPTT - 3; pSrcVMini = pSrcVPTT - 3; }
							if (Diff(pSrcPT + nX2, pSrc + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUPTT - 1; pSrcVMini = pSrcVPTT - 1; }
							if (Diff(pSrcP + nX2, pSrc + nX2, -4, nMiniDiff)) { pSrcUMini = pSrcUP - 2; pSrcVMini = pSrcVP - 2; }
							if (Diff(pSrcPB + nX2, pSrc + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUPBB - 3; pSrcVMini = pSrcVPBB - 3; }
							if (Diff(pSrcPB + nX2, pSrc + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUPBB - 1; pSrcVMini = pSrcVPBB - 1; }

							if (Diff(pSrcT + nX2, pSrc + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUTT - 3; pSrcVMini = pSrcVTT - 3; }
							if (Diff(pSrcT + nX2, pSrc + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUTT - 1; pSrcVMini = pSrcVTT - 1; }
							if (Diff(pSrc + nX2, pSrc + nX2, -4, nMiniDiff)) { pSrcUMini = pSrcU - 2; pSrcVMini = pSrcV - 2; }
							if (Diff(pSrcB + nX2, pSrc + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUBB - 3; pSrcVMini = pSrcVBB - 3; }
							if (Diff(pSrcB + nX2, pSrc + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUBB - 1; pSrcVMini = pSrcVBB - 1; }

							if (Diff(pSrcFT + nX2, pSrc + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUFTT - 3; pSrcVMini = pSrcVFTT - 3; }
							if (Diff(pSrcFT + nX2, pSrc + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUFTT - 1; pSrcVMini = pSrcVFTT - 1; }
							if (Diff(pSrcF + nX2, pSrc + nX2, -4, nMiniDiff)) { pSrcUMini = pSrcUF - 2; pSrcVMini = pSrcVF - 2; }
							if (Diff(pSrcFB + nX2, pSrc + nX2, -6, nMiniDiff)) { pSrcUMini = pSrcUFBB - 3; pSrcVMini = pSrcVFBB - 3; }
							if (Diff(pSrcFB + nX2, pSrc + nX2, -2, nMiniDiff)) { pSrcUMini = pSrcUFBB - 1; pSrcVMini = pSrcVFBB - 1; }

							if (Diff(pSrcP + nX2, pSrc + nX2, -0, nMiniDiff)) { pSrcUMini = pSrcUP - 0; pSrcVMini = pSrcVP - 0; }
							if (Diff(pSrcF + nX2, pSrc + nX2, -0, nMiniDiff)) { pSrcUMini = pSrcUF - 0; pSrcVMini = pSrcVF - 0; }

							if (Diff(pSrcPT + nX2, pSrc + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUPTT + 3; pSrcVMini = pSrcVPTT + 3; }
							if (Diff(pSrcPT + nX2, pSrc + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUPTT + 1; pSrcVMini = pSrcVPTT + 1; }
							if (Diff(pSrcP + nX2, pSrc + nX2, +4, nMiniDiff)) { pSrcUMini = pSrcUP + 2; pSrcVMini = pSrcVP + 2; }
							if (Diff(pSrcPB + nX2, pSrc + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUPBB + 3; pSrcVMini = pSrcVPBB + 3; }
							if (Diff(pSrcPB + nX2, pSrc + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUPBB + 1; pSrcVMini = pSrcVPBB + 1; }

							if (Diff(pSrcT + nX2, pSrc + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUTT + 3; pSrcVMini = pSrcVTT + 3; }
							if (Diff(pSrcT + nX2, pSrc + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUTT + 1; pSrcVMini = pSrcVTT + 1; }
							if (Diff(pSrc + nX2, pSrc + nX2, +4, nMiniDiff)) { pSrcUMini = pSrcU + 2; pSrcVMini = pSrcV + 2; }
							if (Diff(pSrcB + nX2, pSrc + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUBB + 3; pSrcVMini = pSrcVBB + 3; }
							if (Diff(pSrcB + nX2, pSrc + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUBB + 1; pSrcVMini = pSrcVBB + 1; }

							if (Diff(pSrcFT + nX2, pSrc + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUFTT + 3; pSrcVMini = pSrcVFTT + 3; }
							if (Diff(pSrcFT + nX2, pSrc + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUFTT + 1; pSrcVMini = pSrcVFTT + 1; }
							if (Diff(pSrcF + nX2, pSrc + nX2, +4, nMiniDiff)) { pSrcUMini = pSrcUF + 2; pSrcVMini = pSrcVF + 2; }
							if (Diff(pSrcFB + nX2, pSrc + nX2, +6, nMiniDiff)) { pSrcUMini = pSrcUFBB + 3; pSrcVMini = pSrcVFBB + 3; }
							if (Diff(pSrcFB + nX2, pSrc + nX2, +2, nMiniDiff)) { pSrcUMini = pSrcUFBB + 1; pSrcVMini = pSrcVFBB + 1; }
						}

						AverageChroma(pSrcU, pSrcV, pSrcUMini, pSrcVMini, pDestU, pDestV, pEdgeBuffer, nX);
					}
				}
			}

			pSrc += (__int64)nSrcPitch << subSamplingH;
			pSrcP += (__int64)nSrcPitch << subSamplingH;
			pSrcF += (__int64)nSrcPitch << subSamplingH;

			pSrcTT += (__int64)nSrcPitch << subSamplingH;
			pSrcBB += (__int64)nSrcPitch << subSamplingH;
			pSrcPTT += (__int64)nSrcPitch << subSamplingH;
			pSrcPBB += (__int64)nSrcPitch << subSamplingH;
			pSrcFTT += (__int64)nSrcPitch << subSamplingH;
			pSrcFBB += (__int64)nSrcPitch << subSamplingH;

			pSrcT += (__int64)nSrcPitch << subSamplingH;
			pSrcB += (__int64)nSrcPitch << subSamplingH;
			pSrcPT += (__int64)nSrcPitch << subSamplingH;
			pSrcPB += (__int64)nSrcPitch << subSamplingH;
			pSrcFT += (__int64)nSrcPitch << subSamplingH;
			pSrcFB += (__int64)nSrcPitch << subSamplingH;

			pSrcU += nSrcPitchU;
			pSrcUP += nSrcPitchU;
			pSrcUF += nSrcPitchU;
			pSrcV += nSrcPitchU;
			pSrcVP += nSrcPitchU;
			pSrcVF += nSrcPitchU;

			pSrcUTT += nSrcPitchU;
			pSrcUBB += nSrcPitchU;
			pSrcUPTT += nSrcPitchU;
			pSrcUPBB += nSrcPitchU;
			pSrcUFTT += nSrcPitchU;
			pSrcUFBB += nSrcPitchU;
			pSrcVTT += nSrcPitchU;
			pSrcVBB += nSrcPitchU;
			pSrcVPTT += nSrcPitchU;
			pSrcVPBB += nSrcPitchU;
			pSrcVFTT += nSrcPitchU;
			pSrcVFBB += nSrcPitchU;

			pDestU += nDestPitchU;
			pDestV += nDestPitchU;
		}

		free (pEdgeBuffer);

		return destFrame;
	}
};

AVSValue __cdecl Create_DeCross(AVSValue args, void* user_data, IScriptEnvironment* env)
{
	PClip clip = args[0].AsClip();
	const VideoInfo& vi = clip->GetVideoInfo();

	if (vi.BitsPerComponent() != 8 || !(vi.Is420() || vi.Is422())) {
		env->ThrowError("DeCross: Only YV12 and YV16 with constant format and dimensions supported.");
	}

	if ((args[1].AsInt()) < 0 || (args[1].AsInt()) > 255) {
		env->ThrowError("DeCross: thresholdy must be between 0 and 255 (inclusive).");
	}

	if ((args[2].AsInt()) < 0 || (args[2].AsInt()) > 255) {
		env->ThrowError("DeCross: noise must be between 0 and 255 (inclusive).");
	}

	if ((args[3].AsInt()) < 0 || (args[3].AsInt()) > 4) {
		env->ThrowError("DeCross: margin must be between 0 and 4 (inclusive).");
	}

	return new DeCross(args[0].AsClip(), args[1].AsInt(30), args[2].AsInt(60), args[3].AsInt(1), args[4].AsBool(false));
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
	AVS_linkage = vectors;
	env->AddFunction("DeCross", "c[thresholdy]i[noise]i[margin]i[debug]b", Create_DeCross, NULL);
	return "DeCross";
}