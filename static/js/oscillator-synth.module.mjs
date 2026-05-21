import { BaseSynth } from "./base-synth.module.mjs?v=frontend-migration-114";

export class OscillatorSynth extends BaseSynth
{
    // Constructor Method
    constructor(audio_context,master_slider,channel_sliders)
    {
        super(audio_context,master_slider,channel_sliders);
        this.max_voices = 64;
        this.type = "sine";
    }

    // Change Oscillator Type Method
    changeOscType(type)
    {
        this.type = type;
        this.active_nodes.forEach((node) =>
        {
            if(node.subnodes.osc && node.start_time > this.audio_context.currentTime)
            {
                try
                {
                    node.subnodes.osc.type = this.type;
                }
                catch
                {
                    console.log("Couldn't change node oscillator type");
                }
            }
        });
    }

    // Creates Nodes From Event Method
    _createNodeFromEvent(event)
    {
        const osc = this.audio_context.createOscillator();
        osc.type = this.type;
        osc.frequency.value = event.frequency;

        const fade_time = Math.min(0.01,event.duration);
        const osc_level = this.audio_context.createGain();
        osc_level.gain.value = event.amplitude;

        const lfo = this.audio_context.createOscillator();
        lfo.type = "sine";
        lfo.frequency.value = event.rate;
        const lfo_level = this.audio_context.createGain();
        lfo_level.gain.value = event.depth;

        const filter = this.audio_context.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.value = event.cutoff;
        filter.Q.value = event.Q;

        const stereo = this.audio_context.createStereoPanner();
        stereo.pan.value = event.panning;

        // Connect Nodes
        lfo.connect(lfo_level);
        lfo_level.connect(osc.detune);
        osc.connect(filter);
        filter.connect(osc_level);
        osc_level.connect(stereo);

        const node =
        {
            subnodes: {osc,osc_level,lfo,lfo_level,filter,stereo},
            start_time: null,
            stop_time: null,
            connect(dest)
            {
                try
                {
                    stereo.connect(dest);
                }
                catch(error)
                {
                    console.warn("Failed to connect node");
                }
            },
            disconnect()
            {
                try
                {
                    Object.values(this.subnodes).forEach(subnode => subnode.disconnect());
                }
                catch(error)
                {
                    console.warn("Failed to disconnect node");
                }
            },
            start(t)
            {
                try
                {
                    osc_level.gain.setValueAtTime(0,t);
                    osc_level.gain.linearRampToValueAtTime(event.amplitude,t + fade_time);
                    stereo.pan.setValueAtTime(event.panning,t);
                    stereo.pan.linearRampToValueAtTime(-event.panning,t + event.duration);
                    osc.start(t);
                    lfo.start(t);
                    this.start_time = t;
                }
                catch(error)
                {
                    console.warn("Failed to start node");
                    this.start_time = null;
                }
            },
            stop(t)
            {
                try
                {
                    osc_level.gain.setValueAtTime(osc_level.gain.value,t);
                    osc_level.gain.cancelScheduledValues(t);
                    osc_level.gain.linearRampToValueAtTime(0,t + fade_time);
                    osc.stop(t + fade_time);
                    lfo.stop(t + fade_time);
                    this.stop_time = t + fade_time;
                }
                catch(error)
                {
                    console.warn("Failed to stop node");
                    this.stop_time = null;
                }
            }
        }

        return node;
    }
}

window.SoundSketcher.OscillatorSynth = OscillatorSynth;
