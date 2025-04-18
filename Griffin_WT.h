#pragma once
#include <JuceHeader.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "src/griffinwave4/BaseVoiceState.cpp"
#include "src/griffinwave4/BaseVoiceState.h"
#include "src/griffinwave4/Downsampler2Flt.hpp"
#include "src/griffinwave4/rspl.hpp"
#include "src/griffinwave4/InterpFlt.hpp"
#include "src/griffinwave4/InterpPack.cpp"
#include "src/griffinwave4/InterpPack.h"
#include "src/griffinwave4/MipMapFlt.hpp"
#include "src/griffinwave4/ResamplerFlt.cpp"
#include "src/griffinwave4/ResamplerFlt.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace project
{
    using namespace juce;
    using namespace hise;
    using namespace scriptnode;

    template<int NV>
    struct Griffin_WT : data::base
    {
        SNEX_NODE(Griffin_WT);
        struct MetadataClass { SN_NODE_ID("Griffin_WT"); };

        static constexpr bool isModNode() { return false; }
        static constexpr bool isPolyphonic() { return NV > 1; }
        static constexpr bool hasTail() { return false; }
        static constexpr bool isSuspendedOnSilence() { return false; }
        static constexpr int  getFixChannelAmount() { return 2; }
        static constexpr int  NumTables = 0;
        static constexpr int  NumSliderPacks = 0;
        static constexpr int  NumAudioFiles = 0;
        static constexpr int  NumFilters = 0;
        static constexpr int  NumDisplayBuffers = 0;

        Griffin_WT()
            : ready(false),
            pitchParam(0.0),
            frameParam(0),
            volume(0.8f),
            pitchBits(0),
            realLen(0),
            padLen(0),
            sampleLength(0)
        {
        }

        void prepare(PrepareSpecs /*specs*/)
        {
            constexpr int frames = 1000;
            constexpr int frameLen = 2048;
            constexpr int padCycles = 4;

            realLen = frames * frameLen;    // actual loop length
            padLen = padCycles * frameLen; // FIR context
            sampleLength = padLen + realLen;     // total buffer

            sampleBuffer.resize(sampleLength);

            // build original data into temp
            std::vector<float> orig(realLen);
            for (int f = 0; f < frames; ++f)
            {
                float* dst = orig.data() + f * frameLen;
                if (f < 800)
                {
                    // saw wave for first 200 frames
                    double inc = 2.0 / frameLen;
                    double v = -1.0;
                    for (int s = 0; s < frameLen; ++s, v += inc)
                        dst[s] = float(v * 0.8);
                }
                else
                {
                    // sine wave thereafter
                    for (int s = 0; s < frameLen; ++s)
                        dst[s] = float(std::sin(
                            2.0 * M_PI * s / double(frameLen)
                            + f * 0.02
                        ));
                }
            }

            // copy padCycles of the very first frame to pad region
            for (int i = 0; i < padCycles; ++i)
            {
                std::memcpy(
                    sampleBuffer.data() + i * frameLen,
                    orig.data(),
                    frameLen * sizeof(float)
                );
            }

            // copy full real data after pad
            std::memcpy(
                sampleBuffer.data() + padLen,
                orig.data(),
                realLen * sizeof(float)
            );

            // initialize mip?map and resampler on full buffer
            mipMap.init_sample(
                sampleLength,
                rspl::InterpPack::get_len_pre(),
                rspl::InterpPack::get_len_post(),
                12,
                rspl::ResamplerFlt::_fir_mip_map_coef_arr,
                rspl::ResamplerFlt::MIP_MAP_FIR_LEN
            );
            mipMap.fill_sample(sampleBuffer.data(), sampleLength);

            resampler.set_interp(interpPack);
            resampler.set_sample(mipMap);
            resampler.clear_buffers();

            // start playback at first real sample
            resampler.set_pitch(pitchBits);
            rspl::Int64 start = (rspl::Int64)padLen << 32;
            resampler.set_playback_pos(start);

            ready = true;
        }


        void reset()
        {
            resampler.clear_buffers();
        }

        template<typename PD>
        void process(PD& d)
        {
            if (!ready) return;

            auto& fix = d.template as<ProcessData<2>>();
            auto  blk = fix.toAudioBlock();
            float* L = blk.getChannelPointer(0);
            float* R = blk.getChannelPointer(1);
            int    n = d.getNumSamples();

            // fetch current pos
            rspl::Int64 pos = resampler.get_playback_pos();
            rspl::Int64 intPos = pos >> 32;
            rspl::Int64 frac = pos & 0xFFFFFFFF;

            // wrap *only* over realLen, offset by padLen
            rspl::Int64 rel = intPos - padLen;
            rspl::Int64 wrapped = rel & (realLen - 1);
            rspl::Int64 newInt = padLen + wrapped;

            resampler.set_playback_pos((newInt << 32) | frac);

            resampler.interpolate_block(L, n);

            FloatVectorOperations::multiply(L, L, volume, n);
            FloatVectorOperations::copy(R, L, n);
        }

        template<int P>
        void setParameter(double v)
        {
            if constexpr (P == 0)
            {
                pitchParam = v;
                long nb = long(v * (1L << rspl::BaseVoiceState::NBR_BITS_PER_OCT));
                if (nb != pitchBits)
                {
                    pitchBits = nb;
                    resampler.set_pitch(pitchBits);
                }
            }
            else if constexpr (P == 1)
            {
                frameParam = int(v);
            }
            else if constexpr (P == 2)
            {
                volume = float(v);
            }
        }

        void createParameters(ParameterDataList& list)
        {
            {
                parameter::data p("Pitch", { 0.0, 9.0, 0.000001 });
                p.setDefaultValue(0.0);
                registerCallback<0>(p);
                list.add(std::move(p));
            }
            {
                parameter::data p("Frame", { 0.0, 255.0, 1.0 });
                p.setDefaultValue(0.0);
                registerCallback<1>(p);
                list.add(std::move(p));
            }
            {
                parameter::data p("Volume", { 0.0, 1.0, 0.001 });
                p.setDefaultValue(0.8);
                registerCallback<2>(p);
                list.add(std::move(p));
            }
        }

        SN_EMPTY_PROCESS_FRAME;
        void handleHiseEvent(HiseEvent&) {}

    private:
        bool               ready;
        double             pitchParam;
        int                frameParam;
        float              volume;
        long               pitchBits;
        int                realLen, padLen, sampleLength;
        rspl::InterpPack   interpPack;
        rspl::MipMapFlt    mipMap;
        rspl::ResamplerFlt resampler;
        std::vector<float> sampleBuffer;
    };
}
