import { dBToGain } from "./sonification-utils.module.mjs?v=frontend-migration-114";

export class BaseSynth
{
    // Constructor Method
    constructor(audio_context,master_slider,channel_sliders)
    {
        // Audio Context
        this.audio_context = audio_context;

        // Master Slider
        this.master_slider = master_slider;
        this.master_gain = this.audio_context.createGain();
        this.master_gain.connect(this.audio_context.destination);
        this.master_gain.gain.value = dBToGain(this.master_slider.noUiSlider.get(true),this.master_slider.noUiSlider.options.range.min);
        this.master_slider.noUiSlider.on("update",(values) =>
        {
            const value = parseFloat(values[0]);
            this.master_gain.gain.value = dBToGain(value,this.master_slider.noUiSlider.options.range.min);
        });

        // Channel Sliders
        this.channel_gains = [];
        this.channel_sliders = channel_sliders;
        this.channel_sliders.forEach((slider) =>
        {
            const channel_gain = this.audio_context.createGain();
            this.channel_gains.push(channel_gain)
            channel_gain.connect(this.master_gain);
            channel_gain.gain.value = dBToGain(parseFloat(slider.value),slider.min);
            slider.addEventListener("input",(event) => channel_gain.gain.value = dBToGain(parseFloat(event.target.value),slider.min));
        });

        // Member Variables
        this.synth_data = null;
        this.cursor_time = null;
        this.start_time = null;
        this.schedule_index = 0;
        this.schedule_ahead = 0.05;
        this.schedule_interval = 0.02;
        this.is_scheduling = false;
        this.active_nodes = [];
        this.cleanup_nodes = [];
        this.max_voices = 64;
        this.earliest_index = -1;
        this.earliest_stop_time = Number.MAX_SAFE_INTEGER;
    }

    // Prepare To Play Method
    prepareToPlay(synth_data,cursor_time = 0)
    {
        this.stopPlayback();
        this.synth_data = synth_data;
        this.cursor_time = cursor_time;
    }

    // Start Playback Method
    startPlayback()
    {
        this.is_scheduling = true;
        this._scheduleNodes();
    }

    // Stop Playback Method
    stopPlayback()
    {
        this.is_scheduling = false;
        this._clearNodes();
    }

    // Schedule Nodes Method
    _scheduleNodes()
    {
        this.schedule_index = 0;
        this.start_time = this.audio_context.currentTime - this.cursor_time;
        this._schedulerLoop();
    }

    // Scheduler Loop Method
    _schedulerLoop()
    {
        const now = this.audio_context.currentTime;
        const playhead_time = now - this.start_time;
        const window_end = playhead_time + this.schedule_ahead;

        while(this.is_scheduling && this.schedule_index < this.synth_data.length && this.synth_data[this.schedule_index].start_time < window_end)
        {
            const event = this.synth_data[this.schedule_index];
            const start_time = this.start_time + event.start_time;
            const stop_time  = this.start_time + event.end_time;

            if(start_time < now)
            {
                this.schedule_index++;
                continue;
            }

            const node = this._createNodeFromEvent(event);
            if(this.earliest_index == -1)
            {
                this.earliest_stop_time = Number.MAX_SAFE_INTEGER;
                this.active_nodes.forEach((node,index) =>
                {
                    if(node.stop_time < this.earliest_stop_time)
                    {
                        this.earliest_stop_time = node.stop_time;
                        this.earliest_index = index;
                    }
                });
            }

            node.connect(this.channel_gains[event.channel]);
            node.start(start_time);
            node.stop(stop_time);
            this.active_nodes.push(node);

            if(node.stop_time < this.earliest_stop_time)
            {
                this.earliest_index = this.active_nodes.length - 1;
                this.earliest_stop_time = node.stop_time;
            }

            if(this.active_nodes.length > this.max_voices)
            {
                const finishing_node = this.active_nodes[this.earliest_index];
                if(finishing_node.stop_time > start_time)
                {
                    const oldest_node = this.active_nodes.shift();
                    oldest_node.stop(start_time);
                    this.cleanup_nodes.push(oldest_node);
                    this.earliest_index -= 1;
                }
            }
            this.schedule_index++;
        }

        // Remove Inactive Nodes
        let tracked_index = this.earliest_index;
        this.active_nodes = this.active_nodes.filter((node,index) =>
        {
            if(node.stop_time < now)
            {
                node.disconnect();
                if(index === this.earliest_index)
                {
                    tracked_index = -1;
                }
                else if(tracked_index !== -1 && index < this.earliest_index)
                {
                    tracked_index--;
                }
                return false;
            }
            return true;
        });
        this.earliest_index = tracked_index;

        // Schedule Next Loop
        if(this.is_scheduling)
        {
            setTimeout(() => this._schedulerLoop(),this.schedule_interval*1000);
        }
    }

    // Clear Nodes Method
    _clearNodes()
    {
        this.active_nodes.forEach((node) =>
        {
            node.stop(this.audio_context.currentTime);
            node.disconnect();
        });
        this.active_nodes = [];
    }

    // Create Nodes From Event Method
    _createNodeFromEvent(event)
    {
        throw new Error("Subclass must implement this method");
    }
}

window.SoundSketcher.BaseSynth = BaseSynth;
