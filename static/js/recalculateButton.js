//! New script for feature recalculation

document.addEventListener("DOMContentLoaded",() =>
{
    const recalculate_button = document.getElementById('recalculate_button');
    const length_selector = document.getElementById("window_length_selector");
    const overlap_selector = document.getElementById("window_overlap_selector");
    const normalize_button = document.getElementById("normalize_button");
    const filter_button = document.getElementById("filter_button");
    const resketch_button = document.getElementById('submitButton');
    const spinner = document.getElementById('spinner');

    recalculate_button.addEventListener('click',async function()
    {
        console.log("clicked button");
        
        if(globalAudioData)
        {
            spinner.style.display = 'block';

            const n_fft = length_selector.value;
            const overlap = percentages[overlap_selector.value];
            const normalize_audio = normalize_button.checked;
            const apply_filter = filter_button.checked;
            
            const formData = new FormData();
            formData.append('n_fft',n_fft);
            formData.append('overlap',overlap);
            formData.append('normalize_audio',normalize_audio);
            formData.append('apply_filter',apply_filter);
            formData.append('save_json',false);
            formData.append("run_objectifier",false);
            
            const files_processed = globalAudioData.files_processed;
            for(let index = 0; index < files_processed; index++)
            {
                const filename = globalAudioData.filename[index];
                const hash = globalAudioData.hash[index];
                formData.append('filenames',filename);
                formData.append('hashes',hash);
            }

            try
            {
                const response = await fetch("recalculate_features",
                {
                    method: 'POST',
                    body: formData
                });

                if(!response.ok)
                {
                    throw new Error('Upload failed.');
                }
                
                Swal.fire(
                {
                    icon: "success",
                    title: "Audio files processed successfully.",
                    position: "top-end",
                    showConfirmButton: false,
                    timer: 1500
                });

                const data = await response.json();
                globalAudioData = data;
                resketch_button.click();
            }
            catch (err)
            {
                console.error(err);
                alert("Something went wrong during recalculations.");
                spinner.style.display = 'none';
            }
        }
    });
});