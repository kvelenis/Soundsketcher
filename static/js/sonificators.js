// Initialize Synths Method
let synths_initialized = false;
let engineMap = null;
function initSynths()
{
    const osc_master_slider = document.getElementById("oscMaster");
    const grain_master_slider = document.getElementById("grainMaster");
    const channel_sliders = document.querySelectorAll(".volume-slider");
    const oscillator = new OscillatorSynth(audioContext,osc_master_slider,channel_sliders);
    const granulator = new GranularSynth(audioContext,grain_master_slider,channel_sliders);
    engineMap = {"oscillator": oscillator,"granular": granulator};
    synths_initialized = true;
}

let synth_data = [];
function prepareSynthData(data)
{
    synth_data = [];

    const canvas = document.getElementById("svgCanvas");
    const canvasWidth = parseFloat(canvas.getAttribute("width"));
    const canvasPadding = parseInt(canvas.getAttribute("padding"));
    const canvasHeight = parseFloat(canvas.getAttribute("height"));
    const maxDuration = canvasWidth/pixelsPerSecond;

    const width_slider = document.getElementById("slider-2");
    const min_width = width_slider.noUiSlider.options.range.min;
    const max_width = width_slider.noUiSlider.options.range.max;

    const angle_slider = document.getElementById("slider-4");
    const max_angle = angle_slider.noUiSlider.options.range.max*Math.PI/180;

    const dash_slider = document.getElementById("slider-7");
    const min_dash = dash_slider.noUiSlider.options.range.min;
    const max_dash = dash_slider.noUiSlider.options.range.max;

    let id = -1;
    for(channel in data)
    {
        const channel_data = data[channel];
        channel_data.forEach((path) =>
        {
            // Event ID
            id++;
            
            // Path Data
            const timestamp = path.timestamp;
            const y_value = path.yAxis;
            const length = path.lineLength;
            const width = path.lineWidth;
            const hue = path.colorHue;
            const saturation = path.colorSaturation;
            const lightness = path.colorLightness;
            const angle = path.angle;
            const dash = path.dashArray;
    
            // Timestamps
            const half_duration = ((length/2)*Math.abs(Math.cos(angle)) + (width/2))/pixelsPerSecond;
            const start_time = clamp(timestamp - half_duration,0,maxDuration);
            const end_time = clamp(timestamp + half_duration,0,maxDuration);
            const duration = end_time - start_time;

            // Panning
            const panning = map(angle - Math.PI/2,-max_angle,max_angle,1,-1);

            // Amplitude
            const min_db = -30;
            const max_db = 0;
            const db = map(width,min_width,max_width,min_db,max_db);
            const amplitude = width === 0 ? 0 : Math.pow(10,db/20);

            // Detune
            const min_detune = 0;
            const max_detune = getRandomSign()*50;
            const detune_cents = map(saturation,0,100,min_detune,max_detune);
            const detune = Math.pow(2,(detune_cents/100)/12);
    
            // Frequency
            const min_pitch = 36;
            const max_pitch = 96;
            const y_normalized = (y_value - canvasPadding)/(canvasHeight - 2*canvasPadding);
            const pitch = Math.round(map(y_normalized,0,1,max_pitch,min_pitch));
            const base_frequency = 440*Math.pow(2,(pitch - 69)/12);
            const frequency = base_frequency*detune;
    
            // Cutoff
            const min_cutoff = clamp(frequency/3,20,20000);
            const max_cutoff = clamp(frequency*30,20,20000);
            const cutoff = map(lightness,0,100,min_cutoff,max_cutoff);
    
            // Resonance
            const min_Q = 0.5;
            const max_Q = 20;
            const [r,g,b] = hslToRgb(hue,100,50);
            const brightness = perceivedBrightness(r,g,b);
            const Q = map(brightness,18,237,min_Q,max_Q);

            // LFO
            const min_rate = 5;
            const max_rate = 20;
            const rate = map(dash,min_dash,max_dash,max_rate,min_rate);
            const min_depth = 0;
            const max_depth = 50;
            const depth = map(dash,min_dash,max_dash,min_depth,max_depth);

            // Grain Playback Rate
            const min_playback_rate = 0.5;
            const max_playback_rate = 2;
            const playback_rate = map(y_normalized,0,1,max_playback_rate,min_playback_rate);

            // Grain Density
            const min_density = 1/duration;
            const max_density = 5/duration;
            const density = map(dash,min_dash,max_dash,min_density,max_density);

            // Grain Size
            const min_size = 1/density;
            const max_size = duration;
            const size = map(lightness,0,100,min_size,max_size);

            // Grain Spread
            const min_spread = 0;
            const max_spread = 0.25;
            const spread = map(saturation,0,100,max_spread,min_spread);

            // Grain Source Position
            const source_position = hue/360; 

            synth_data.push({id,channel,start_time,end_time,duration,amplitude,frequency,cutoff,Q,panning,rate,depth,playback_rate,size,density,spread,source_position});
        });
    }
    synth_data.sort((a,b) => a.start_time - b.start_time);
}

// --------------------------------------------------
// Base Synth Class
// --------------------------------------------------
class BaseSynth
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
        this.master_gain.gain.value = parseFloat(this.master_slider.value);
        this.master_slider.addEventListener("input",(event) => this.master_gain.gain.value = parseFloat(event.target.value));

        // Channel Sliders
        this.channel_gains = [];
        this.channel_sliders = channel_sliders;
        this.channel_sliders.forEach((slider) =>
        {
            const channel_gain = this.audio_context.createGain();
            this.channel_gains.push(channel_gain)
            channel_gain.connect(this.master_gain);
            channel_gain.gain.value = parseFloat(slider.value);
            slider.addEventListener("input",(event) => channel_gain.gain.value = parseFloat(event.target.value));
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

        console.log("Active Voices: ",this.active_nodes.length); //!
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

class OscillatorSynth extends BaseSynth
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
        const now = this.audio_context.currentTime;
        this.active_nodes.forEach((node) =>
        {
            if(node.subnodes.osc && node.start_time > now)
            {
                node.subnodes.osc.type = this.type;
            }
        });
    }

    // Creates Nodes From Event Method
    _createNodeFromEvent(event)
    {
        const osc = this.audio_context.createOscillator();
        osc.type = this.type;
        osc.frequency.value = event.frequency;

        const fade_time = Math.min(0.01,event.duration/2);
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

class GranularSynth extends BaseSynth
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
        const now = this.audio_context.currentTime;
        this.active_nodes.forEach((node) =>
        {
            const grain_nodes = node.subnodes;
            if(grain_nodes && node.start_time > now)
            {
                grain_nodes.forEach((grain) =>
                {
                    grain.subnodes.source.buffer = buffer;
                    grain.position = grain.position/this.buffer.duration*buffer.duration;
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

            const fade_time = Math.min(0.01,grain_duration*0.1);

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
                        source.start(t + grain_offset,this.position,grain_duration*event.playback_rate);
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

// --------------------------------------------------
// Oscillator Engine UI
// --------------------------------------------------
document.addEventListener("DOMContentLoaded",() =>
{
    // DOM Elements
    const waveformImage = document.getElementById("waveformImage");
    const waveformPrev = document.getElementById("waveformPrev");
    const waveformNext = document.getElementById("waveformNext");

    // Waveform Selection
    const waveforms =
    [
        { name: "sine", image: "static/assets/sine-wave.svg" },
        { name: "triangle", image: "static/assets/triangle-wave.svg" },
        { name: "square", image: "static/assets/square-wave.svg" },
        { name: "sawtooth", image: "static/assets/sawtooth-wave.svg" },
    ];
    let currentWaveformIndex = 0;

    // Waveform Update Function
    function updateWaveform()
    {
        const currentWaveform = waveforms[currentWaveformIndex];
        const selectedWaveform = currentWaveform.name;
        waveformImage.src = currentWaveform.image;
        waveformImage.alt = selectedWaveform.charAt(0).toUpperCase() + selectedWaveform.slice(1) + " Wave";
        if(synths_initialized)
        {
            engineMap["oscillator"].changeOscType(selectedWaveform);
        }
    }

    // Initial Waveform Update
    updateWaveform();

    // Waveform Update Listeners
    waveformPrev.addEventListener("click",() =>
    {
        currentWaveformIndex = (currentWaveformIndex - 1 + waveforms.length) % waveforms.length;
        updateWaveform();
    });
    waveformNext.addEventListener("click",() =>
    {
        currentWaveformIndex = (currentWaveformIndex + 1) % waveforms.length;
        updateWaveform();
    });

    // Play Button
    const playButton = document.getElementById("playOscillator");
    playButton.addEventListener("click",() =>
    {
        if(synths_initialized)
        {
            const osc_type = waveforms[currentWaveformIndex].name;
            const oscillator = engineMap["oscillator"]; 
            if(oscillator.type !== osc_type)
            {
                oscillator.changeOscType(osc_type);
            }
        }
    });
});

// --------------------------------------------------
// Granular Engine UI
// --------------------------------------------------
document.addEventListener("DOMContentLoaded",() =>
{
    let file_data = null;
    let file_uploaded = false;
    let sample_buffer = null;

    // DOM Elements
    const header = document.getElementById("granular-engine-header")
    const container = document.querySelector(".waveform-container");
    const canvas = document.getElementById("waveformCanvas");
    const file_input = document.getElementById('fileInput');
    const text = document.getElementById('waveform-text');
    const spinner = document.getElementById('spinner');
    const resketch_button = document.getElementById("submitButton");

    // Resize Canvas Function
    function resizeCanvas()
    {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        if(canvas.width > 0 && canvas.height > 0)
        {
            drawWaveform();
        }
    }

    // Resize Canvas Listeners
    header.addEventListener("click",resizeCanvas);
    window.addEventListener("resize",resizeCanvas);
    resketch_button.addEventListener('click',resizeCanvas);

    // Drag & Drop Event Listeners
    container.addEventListener("dragenter",(event) =>
    {
        event.preventDefault();
        container.classList.add('solid');
    });
    container.addEventListener("dragover",(event) =>
    {
        event.preventDefault();
        container.classList.add("solid");
    });
    container.addEventListener("dragleave",(event) =>
    {
        event.preventDefault();
        if(!file_uploaded)
        {
            container.classList.remove('solid');
        }
    });
    container.addEventListener("drop",(event) =>
    {
        event.preventDefault();
        uploadFile(event.dataTransfer.files);
    });
        
    // File Input Listeners
    inputClickWrapper = () => file_input.click();
    container.addEventListener('click',inputClickWrapper);
    file_input.addEventListener('change',(event) =>
    {
        event.preventDefault();
        uploadFile(event.target.files);
        file_input.value = '';
    });

    // Upload File Function
    async function uploadFile(files)
    {
        // Prepare UI Elements
        spinner.style.display = 'block';
        container.classList.add('solid');

        // Select Audio File
        const valid_files = Array.from(files).filter(file => file.type.startsWith('audio/'));
        if(valid_files.length === 0)
        {
            alert('No valid audio files were dropped.');
            spinner.style.display = 'none';
            if(!file_uploaded)
            {
                container.classList.remove('solid');
            }
            return;
        }
        const file = valid_files[0];

        // Load Audio Data
        try
        {
            const arrayBuffer = await file.arrayBuffer();
            sample_buffer = await audioContext.decodeAudioData(arrayBuffer);
            processBuffer(sample_buffer);
            file_uploaded = true;    
            if(synths_initialized)
            {
                engineMap["granular"].changeBuffer(sample_buffer);
            }
        }
        catch
        {
            alert('Failed to load audio data.');
            spinner.style.display = 'none';
            if(!file_uploaded)
            {
                container.classList.remove('solid');
            }
            return;
        }
          
        // Update UI Elements
        text.style.display = "none";
        drawWaveform(file_data);
        spinner.style.display = 'none';
        // container.removeEventListener('click',inputClickWrapper);
        // container.style.cursor = 'default';
    }

    function processBuffer(buffer)
    {
        // Retrieve Buffer Channels
        const left = buffer.getChannelData(0);
        const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
        
        // Convert to Mono
        const length = buffer.length;
        file_data = new Float32Array(length);
        for(let i = 0; i < length; i++)
        {
            file_data[i] = (left[i] + right[i])/2;
        }

        // Normalize Data
        let max = 0;
        for(let i = 0; i < length; i++)
        {
            const absVal = Math.abs(file_data[i]);
            if(absVal > max)
            {
                max = absVal;
            }
        }
        if(max !== 0)
        {
            for(let i = 0; i < length; i++)
            {
                file_data[i] /= max;
            }
        }
    }

    // Draw Waveform Function
    function drawWaveform()
    {
        // Check Data Validity
        if(!file_data)
        {
            return;
        }

        // Initialize Canvas
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0,0,canvasWidth,canvasHeight);

        // Draw Y Center
        const y_center = canvasHeight/2;
        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.strokeStyle = getComputedStyle(container).borderColor;
        ctx.moveTo(0,y_center);
        ctx.lineTo(canvasWidth,y_center);
        ctx.stroke();
        
        // Draw Waveform
        const num_samples = file_data.length;
        if(num_samples <= canvasWidth/2)
        {
            // Set Colors
            ctx.strokeStyle = "rgba(0, 0, 128, 1)";
            ctx.fillStyle = ctx.strokeStyle;

            // Calculate Step Size
            const step = canvasWidth/num_samples;
            let x = step/2;

            // Draw Samples
            for(let i = 0; i < num_samples; i++)
            {
                // Calculate Y Coordinate
                const sample = file_data[i];
                const y = (1 - sample)*y_center;

                // Draw Line
                ctx.beginPath();
                ctx.moveTo(x,y_center);
                ctx.lineTo(x,y);
                ctx.stroke();

                // Draw Circle
                ctx.beginPath();
                ctx.arc(x,y,2,0,Math.PI*2);
                ctx.fill();

                // Update Step
                x += step;
            }
        }
        else
        {
            // Set Colors
            ctx.strokeStyle = "rgba(0, 0, 128, 1)";
            ctx.fillStyle = "rgba(0, 96, 255, 1)";

            // Compute Envelope
            const envelope = new Array(canvasWidth);
            const step = Math.ceil(num_samples/canvasWidth);
            for (let x = 0; x < canvasWidth; x++)
            {
                let min = Infinity;
                let max = -Infinity;
                const start = Math.min(x*step,num_samples - 1);
                const end = Math.min(start + step,num_samples);
                for(let j = start; j < end; j++)
                {
                    const sample = file_data[j];
                    min = Math.min(min,sample);
                    max = Math.max(max,sample);
                }
                envelope[x] = {min,max};
            }

            // Draw Envelope
            const path = new Path2D();
            const x0 = 0;
            const y0 = (1 - envelope[x0].max)*y_center;
            path.moveTo(x0,y0);
            for(let x = 0; x < canvasWidth; x++)
            {
                const y = (1 - envelope[x].max)*y_center;
                path.lineTo(x,y);
            }
            for(let x = canvasWidth - 1; x >= 0; x--)
            {
                const y = (1 - envelope[x].min)*y_center;
                path.lineTo(x,y);
            }
            path.closePath();
            ctx.fill(path);
            ctx.stroke(path);
        }

        // Draw Source Positions
        const color_sliders = document.querySelectorAll(".color-slider");
        if (color_sliders)
        {
            color_sliders.forEach((slider) =>
            {
                const hue = slider.value;
                const x = canvasWidth*hue/360;
                ctx.strokeStyle = `hsl(${hue},100%,50%)`;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(x,0);
                ctx.lineTo(x,canvasHeight);
                ctx.stroke();
            });
        }
    }

    // Play Button
    const playButton = document.getElementById("playGranulator");
    playButton.addEventListener("click",(event) =>
    {
        if(!file_data)
        {
            event.stopImmediatePropagation();
            alert("Please load an audio file first!");
            return
        }

        if(synths_initialized)
        {
            const granulator = engineMap["granular"]; 
            if(granulator.buffer !== sample_buffer)
            {
                granulator.changeBuffer(sample_buffer);
            }
        }
    });
});

// --------------------------------------------------
// Shared Logic
// --------------------------------------------------
document.addEventListener("DOMContentLoaded",() =>
{
    let synth = null;
    let animation = null;
    let currentActiveButton = null;

    // DOM Elements
    const playButtons = document.querySelectorAll(".sonificators__play-button");
    const playStopButton = document.getElementById("playStopButtonHeader");
    const playStopIcon = document.getElementById("playStopIconHeader");
    const resketch_button = document.getElementById("submitButton");
    const polygonMode = document.getElementById("linePolygonMode");
    const objectifierMode = document.getElementById("objectifierMode");

    // Start Playback Function
    async function startPlayback()
    {
        if(audioContext.state === "suspended")
        {
            await audioContext.resume();
        }
        synth.prepareToPlay(synth_data,cursorTime)
        synth.startPlayback();
        startCursorAnimation(!synthPlaying);
        playStopIcon.src = "static/assets/stop.png";
        playStopIcon.alt = "Stop";
        synthPlaying = true;
    }

    // Stop Playback Function
    function stopPlayback()
    {
        synth.stopPlayback()
        stopCursorAnimation();
        playStopIcon.src = "static/assets/play.png";
        playStopIcon.alt = "Play";
        synthPlaying = false;
        synth = null;
    }

    // Update Buttons Function
    function updateButtons(newActiveButton = null)
    {
        currentActiveButton = newActiveButton;
        playButtons.forEach((button) =>
        {
            const isActive = (button === currentActiveButton);
            button.textContent = isActive ? "Stop" : "Play";
        });
    }

    // Set up Playback for Buttons
    playButtons.forEach((button) =>
    {
        button.addEventListener("click",() =>
        {
            // Toggle Button Off
            const wasActive = (button === currentActiveButton);
            if(wasActive)
            {
                stopPlayback();
                updateButtons();
                return;
            }

            // Stop Audio File Playback
            if(isPlaying)
            {
                togglePlayStop();
            }

            // Toggle Button On
            if(Object.keys(pathData).length > 0)
            {
                if(polygonMode.checked || objectifierMode.checked)
                {
                    alert("Sonificators are currently not available in Polygon and Objectifier modes");
                    return;
                }
                if(synthPlaying)
                {
                    cursorTime = parseFloat(document.getElementById("progressLine").getAttribute("x1"))/pixelsPerSecond;
                    synth.stopPlayback()
                }
                synth = engineMap[button.dataset.engine];
                startPlayback();
                updateButtons(button);
            }
        });
    });

    // Handle Playback on Resketch Button Click
    resketch_button.addEventListener('click',() =>
    {
        // Stop Synth Audio Playback
        if(synthPlaying)
        {
            stopPlayback();
            updateButtons();
        }

        // Handle Playback on SVG Canvas Click
        const svgCanvas = document.getElementById("svgCanvas");
        svgCanvas.addEventListener("click",(event) =>
        {
            const rect = svgCanvas.getBoundingClientRect();
            cursorTime = (event.clientX - rect.left)/pixelsPerSecond;
            if(synthPlaying)
            {
                startPlayback();
            }
        });
    });

    // Handle Playback on Play/Stop Button Click
    playStopButton.addEventListener("click",() =>
    {
        if(synthPlaying)
        {
            stopPlayback();
            updateButtons();
        }
    });

    // Handle Playback on Space Keydown
    document.addEventListener("keydown",(event) =>
    {
        if(event.code === "Space") 
        {
            if(synthPlaying)
            {
                stopPlayback();
                updateButtons();
            }
        }
    });

    // Draw Cursor Function
    function drawCursor(x = 0)
    {
        let cursor = document.getElementById("progressLine");
        if(!cursor)
        {
            cursor = document.createElementNS("http://www.w3.org/2000/svg","line");
            cursor.setAttribute("id","progressLine");
            cursor.setAttribute("y1",0);
            cursor.setAttribute("y2",svgCanvas.getAttribute("height"));
            cursor.setAttribute("stroke","red");
            cursor.setAttribute("stroke-width",2);
            svgCanvas.appendChild(cursor);
        }
        cursor.setAttribute("x1",x);
        cursor.setAttribute("x2",x);
    }

    // Start Cursor Animation Function
    function startCursorAnimation(init = true)
    {
        const padding = 100;
        const svgWrapper = document.getElementById("svgWrapper");
        const isScrollableMode = document.getElementById("scrollModeToggle").checked;
        const canvasWidth = Number(document.getElementById("svgCanvas").getAttribute("width"));

        // Calculate Start Time
        const startTime = audioContext.currentTime - cursorTime;

        // Stop Previous Animation
        stopCursorAnimation();

        // Update Function
        function update()
        {
            // Draw Cursor
            const elapsedTime = audioContext.currentTime - startTime;
            const x_canvas = elapsedTime*pixelsPerSecond;
            drawCursor(x_canvas);

            // Scroll Canvas
            if(isScrollableMode)
            {
                const x_screen = x_canvas - svgWrapper.scrollLeft;
                if(x_screen > padding)
                {
                    svgWrapper.scrollLeft = x_canvas - padding;
                }
                if(init)
                {
                    svgWrapper.scrollLeft = x_canvas;
                    init = false;
                }
            }
    
            // Stop Playback
            if(x_canvas >= canvasWidth)
            {
                updateButtons();
                stopPlayback();
                return;
            }

            // Update Animation
            animation = requestAnimationFrame(update);
        }

        // Start Animation
        update();
    }

    // Stop Cursor Animation Function
    function stopCursorAnimation()
    {
        if(animation !== null)
        {
            cancelAnimationFrame(animation);
            animation = null;
            const cursor = document.getElementById("progressLine");
            if(cursor)
            {
                cursor.remove();
            }
            cursorTime = 0;
        }
    }
});