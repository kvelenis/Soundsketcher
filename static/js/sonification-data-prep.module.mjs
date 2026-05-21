import { clamp, getRandomSign, hslToRgb, map, perceivedBrightness } from "./sonification-utils.module.mjs?v=frontend-migration-114";

function getSonificationStateForDataPrep() {
    return window.SoundSketcher.sonificationState;
}

export function prepareSynthData(data)
{
    const sonificationState = getSonificationStateForDataPrep();
    sonificationState.synthData = [];

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
    for(let channel in data)
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
            const min_db = -18;
            const max_db = 12;
            const db = map(width,min_width,max_width,min_db,max_db);
            const amplitude = (width === 0) ? 0 : Math.pow(10,db/20);

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

            sonificationState.synthData.push({id,channel,start_time,end_time,duration,amplitude,frequency,cutoff,Q,panning,rate,depth,playback_rate,size,density,spread,source_position});
        });
    }
    sonificationState.synthData.sort((a,b) => a.start_time - b.start_time);
}

window.SoundSketcher.sonificationDataPrep = {
    prepareSynthData,
};
window.prepareSynthData = prepareSynthData;
