import { BaseSynth } from "./base-synth.module.mjs?v=frontend-migration-114";
import { clamp } from "./sonification-utils.module.mjs?v=frontend-migration-114";

export class GranularSynth extends BaseSynth
{
    // Constructor Method
    constructor(audio_context,master_slider,channel_sliders)
    {
        super(audio_context,master_slider,channel_sliders);
        this.max_voices = 128;
        this.buffer = null;
        this.rand = null;
        this._resetRandomEngine(0);
    }

    _resetRandomEngine(seed)
    {
        this.rand = function()
        {
            let t = seed += 0x6D2B79F5;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        }
    }

    // Change Buffer Method
    changeBuffer(buffer)
    {
        this.active_nodes.forEach((node) =>
        {
            const grain_nodes = node.subnodes;
            if(grain_nodes && node.start_time > this.audio_context.currentTime)
            {
                grain_nodes.forEach((grain) =>
                {
                    if(grain.start_time > this.audio_context.currentTime && grain.subnodes.source.buffer !== buffer)
                    {
                        try
                        {
                            grain.subnodes.source.buffer = buffer;
                            grain.position = grain.position/this.buffer.duration*buffer.duration;
                        }
                        catch
                        {
                            console.log("Couldn't change node source buffer");
                        }
                    }
                });
            }
        });
        this.buffer = buffer;
    }

    // Create Nodes From Event Method
    _createNodeFromEvent(event)
    {
        this._resetRandomEngine(event.id);
        const grain_nodes = [];
        const num_grains = Math.ceil(event.duration*event.density);
        for(let index = 0; index < num_grains; index++)
        {
            const grain_offset = index/event.density;
            const grain_duration = Math.min(event.size,event.duration - grain_offset);
            const grain_position = clamp((event.source_position + (this.rand() - 0.5)*event.spread),0,1)*this.buffer.duration;

            const source = this.audio_context.createBufferSource();
            source.buffer = this.buffer;
            source.playbackRate.value = event.playback_rate;
            source.loop = true;

            const fade_time = Math.min(0.01,grain_duration);

            const level = this.audio_context.createGain();
            level.gain.value = event.amplitude;

            const stereo = this.audio_context.createStereoPanner();
            stereo.pan.value = event.panning;

            // Connect Nodes
            source.connect(level);
            level.connect(stereo);

            const grain_node =
            {
                subnodes: {source,level,stereo},
                start_time: null,
                stop_time: null,
                position: grain_position,
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
                        // Start Grain
                        level.gain.setValueAtTime(0,t + grain_offset);
                        level.gain.linearRampToValueAtTime(event.amplitude,t + grain_offset + fade_time);
                        stereo.pan.setValueAtTime(event.panning,t);
                        stereo.pan.linearRampToValueAtTime(-event.panning,t + event.duration);
                        source.start(t + grain_offset,this.position,(grain_duration + fade_time)*event.playback_rate);
                        this.start_time = t + grain_offset;

                        // Stop Grain
                        level.gain.setValueAtTime(level.gain.value,this.start_time + grain_duration);
                        level.gain.cancelScheduledValues(this.start_time + grain_duration);
                        level.gain.linearRampToValueAtTime(0,this.start_time + grain_duration + fade_time);
                        source.stop(this.start_time + grain_duration + fade_time);
                        this.stop_time = this.start_time + grain_duration + fade_time;
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
                        level.gain.setValueAtTime(level.gain.value,t);
                        level.gain.cancelScheduledValues(t);
                        level.gain.linearRampToValueAtTime(0,t + fade_time);
                        source.stop(t + fade_time);
                        this.stop_time = t + fade_time;
                    }
                    catch(error)
                    {
                        console.warn("Failed to stop node");
                        this.stop_time = null;
                    }
                }
            }
            grain_nodes.push(grain_node);
        }

        const node =
        {
            subnodes: grain_nodes,
            start_time: null,
            stop_time: null,
            connect(dest)
            {
                try
                {
                    this.subnodes.forEach(subnode => subnode.connect(dest));
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
                    this.subnodes.forEach(subnode => subnode.disconnect());
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
                    this.subnodes.forEach(subnode => subnode.start(t));
                    if(this.subnodes.length > 0)
                    {
                        this.start_time = Math.min(...this.subnodes.map(subnode => subnode.start_time));
                    }
                    else
                    {
                        this.start_time = null;
                    }
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
                    this.subnodes.forEach(subnode => subnode.stop(t));
                    if(this.subnodes.length > 0)
                    {
                        this.stop_time  = Math.max(...this.subnodes.map(subnode => subnode.stop_time));
                    }
                    else
                    {
                        this.stop_time = null;
                    }
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

window.SoundSketcher.GranularSynth = GranularSynth;
