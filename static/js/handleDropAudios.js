// Compute SHA-256 hash of file
function computeSHA256(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = function () {
            const shaObj = new jsSHA("SHA-256", "ARRAYBUFFER");
            shaObj.update(reader.result);
            const hash = shaObj.getHash("HEX");
            resolve(hash);
        };

        reader.onerror = () => reject(reader.error);

        reader.readAsArrayBuffer(file);
    });
}



// Ask backend if hashed file exists
async function checkFileExists(hash) {
    const response = await fetch(`check_file_exists?audio_hash=${hash}`);
    if (!response.ok) return false;
    const result = await response.json();
    return result.features_exists;
}


// async function handleDrop(event) {
//     event.preventDefault();
//     const spinner = document.getElementById('spinner');
//     spinner.style.display = 'block';

//     const files = Array.from(event.dataTransfer.files).filter(file => file.type === 'audio/wav');
//     if (files.length === 0) {
//         alert('No valid WAV files were dropped.');
//         spinner.style.display = 'none';
//         return;
//     }

//     const formData = new FormData();
//     let atLeastOneCached = false;
//     const hashes = {};

//     // First pass: check hashes and determine if any cache exists
//     for (const file of files) {
//         const hash = await computeSHA256(file);
//         hashes[file.name] = hash;

//         const exists = await checkFileExists(hash);
//         if (exists) {
//             atLeastOneCached = true;
//         }

//         formData.append('wav_files', file); // Always append, let backend decide
//     }

//     // If none of the files are cached, skip asking
//     let reuseCached = false;
//     if (atLeastOneCached) {
//         const reuseChoice = await Swal.fire({
//             title: "Use cached analysis if available?",
//             text: "Some files have already been analyzed. Reuse or reprocess?",
//             icon: "question",
//             showCancelButton: true,
//             confirmButtonText: "Use Cached",
//             cancelButtonText: "Reprocess All"
//         });
//         reuseCached = reuseChoice.isConfirmed;
//     }

//     try {
//         const response = await fetch(`/upload_wavs?reuse_cached=${reuseCached}&run_objectifier=true`, {
//             method: 'POST',
//             body: formData
//         });

//         if (!response.ok) throw new Error('Upload failed.');
//         const data = await response.json();

//         dropArea.style.display = "none";
//         Swal.fire({
//             icon: "success",
//             title: "WAV files processed successfully.",
//             position: "top-end",
//             showConfirmButton: false,
//             timer: 1500
//         });

//         globalAudioData = data;
//         globalFile = files;
//         visualizeAllFiles(data, files);

//     } catch (err) {
//         console.error(err);
//         alert("Something went wrong during upload.");
//     } finally {
//         spinner.style.display = 'none';
//     }
// }

//! Declared these explicitly here
globalAudioData = null;
globalFile = null;

//! Light modifications
async function handleDrop(event) {
    event.preventDefault();

    if(!isLoading)
    {

        const spinner = document.getElementById('spinner');
        spinner.style.display = 'block';
        isLoading = true;
    
        const n_fft = document.getElementById("window_length_selector").value;
        const overlap = percentages[document.getElementById("window_overlap_selector").value];
        const normalize_audio = document.getElementById("normalize_button").checked;
        const apply_filter = document.getElementById("filter_button").checked;
    
        // Accept all audio types, not just .wav
        const files = Array.from(event.dataTransfer.files).filter(file => file.type.startsWith('audio/'));
        if (files.length === 0) {
            alert('No valid audio files were dropped.');
            spinner.style.display = 'none';
            dropArea.classList.remove('solid');
            return;
        }
    
        const formData = new FormData();
        let atLeastOneCached = false;
    
        // First pass: check hashes and determine if any cache exists
        for (const file of files) {
    
            const hash = await computeSHA256(file);
            const exists = await checkFileExists(hash);
            if(exists)
            {
                atLeastOneCached = true;
            }
    
            // Append with a generic field name the backend expects (you can adjust this)
            formData.append('audio_files', file);
    
            //! Added this here
            formData.append('filenames',file.name);
            formData.append('hashes',hash);
            formData.append('n_fft',n_fft);
            formData.append('overlap',overlap);
            formData.append('normalize_audio',normalize_audio);
            formData.append('apply_filter',apply_filter);
            formData.append('save_json',true);
            formData.append("run_objectifier",false);
        }
    
        let reuseCached = false;
        if (atLeastOneCached) {
            const reuseChoice = await Swal.fire({
                title: "Use cached analysis if available?",
                text: "Some files have already been analyzed. Reuse or reprocess?",
                icon: "question",
                showCancelButton: true,
                confirmButtonText: "Use Cached",
                cancelButtonText: "Reprocess All"
            });
            reuseCached = reuseChoice.isConfirmed;
        }
    
        try
        {
            const response = await fetch(`upload_wavs?reuse_cached=${reuseCached}`,
            {
                method: 'POST',
                body: formData
            });
    
            if (!response.ok) throw new Error('Upload failed.');
            const data = await response.json();
    
            dropArea.style.display = "none";
            
            globalAudioData = data;
            globalFile = files;
            await visualizeAllFiles(globalFile); //! Changed arguments for new version of function
    
            //! Moved this here
            Swal.fire({
                icon: "success",
                title: "Audio files processed successfully.",
                position: "top-end",
                showConfirmButton: false,
                timer: 1500
            });
    
        } catch (err) {
            console.error(err);
            alert("Something went wrong during upload.");
            dropArea.classList.remove('solid');
        } finally {
            spinner.style.display = 'none';
            isLoading = false;
        }
    }
    else
    {
        alert("Please wait for loading to finish");
    }
}


/**
 * Returns a distinct base hue for a given file index.
 * Loops if there are more files than colors.
 */
// Changed this a little
// function getFileBaseHue(fileIndex) {
//     const fileBaseHues = [
//       0,   // Red
//       120, // Green
//       240, // Blue
//       300, // Magenta
//       180, // Cyan
//       60,  // Yellow
//     ];
//     // const fileBaseHues = [
//     //   0,    // Red
//     //   120,  // Green
//     //   240,  // Blue
//     //   45,   // Orange/Yellow
//     //   300   // Magenta
//     // ];
//     return fileBaseHues[fileIndex % fileBaseHues.length];
//   }
  
//! New version 'visualizeAllFiles'
async function visualizeAllFiles(fileList)
{
    document.getElementById("mixer-subcontainer").style.display = "flex";
    console.log("PRE PRE fileList: ",fileList);
    if (fileList && fileList.length > 0) {
        console.log("PRE fileList", fileList);
        if (fileList[0] instanceof File) {
            console.log("WE ARE HERE");
            await loadAudioFiles(fileList); //! Added await
        } else if (fileList[0].audio_url) {
            console.log("fileList:  ",fileList);
            await loadAudioFromUrls(fileList); //! Added await
        }
    }
    createMixerUI();
    initSynths();
    const resketch_button = document.getElementById('submitButton');
    resketch_button.click()
}

//! Previous version 'visualizeAllFiles'
// async function visualizeAllFiles(audioData, fileList) {

//     const existingSVG = document.getElementById("svgCanvas");
//     if (existingSVG) {
//         existingSVG.remove();
//     }
//     // const canvasWidth = 2000;
//     const canvasWidth = window.innerWidth;

//     const svgContainer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
//     svgContainer.setAttribute("id", "svgCanvas");
//     svgContainer.setAttribute("width", canvasWidth);
//     svgContainer.setAttribute("height", 800);

//     document.body.appendChild(svgContainer);

//     document.getElementById("controls").style.display = "flex";

//     const maxDuration = Math.max(
//         ...audioData.features.map(f =>
//             f.features[f.features.length - 1]?.timestamp || 0
//         )
//     );

//     audioData.features.forEach((fileData, fileIndex) => {
//         console.log("fileIndex", fileIndex, "fileData:", fileData);
//         const baseHue = getFileBaseHue(fileIndex);
//         drawVisualization(fileData, svgContainer, canvasWidth, maxDuration, fileIndex, baseHue, false);
//         //drawClusterOverlays(fileData.clusters, svgContainer, canvasWidth, 800, maxDuration);
//     });

//     document.getElementById("svgCanvas").addEventListener("click", (event) => {
//         const svgCanvas = event.currentTarget;
//         const boundingRect = svgCanvas.getBoundingClientRect();
//         const clickX = event.clientX - boundingRect.left;
//         clickedPosition = (clickX / boundingRect.width) * maxDuration;
//         clickedPosition = Math.max(0, Math.min(clickedPosition, maxDuration));
//         moveVerticalLine(clickX);
//     });

//     //! Moved this here
//     console.log("PRE PRE fileList: ",fileList);
//     if (fileList && fileList.length > 0) {
//         console.log("PRE filelist", fileList);
//         if (fileList[0] instanceof File) {
//             console.log("WE ARE HERE");
//             await loadAudioFiles(fileList); //! Added await
//         } else if (fileList[0].audio_url) {
//             console.log("fileList:  ",fileList);
//             await loadAudioFromUrls(fileList); //! Added await
//         }
//     }
//     createMixerUI();

//     initializeTooltip();
// }

//! Light modifications
async function fetchPreviouslyProcessed(filename, hash) {
    
    if(!isLoading)
    {
        //! Added this
        const spinner = document.getElementById('spinner');
        spinner.style.display = 'block';
        isLoading = true;
    
        const formData = new FormData();
        formData.append("filename", filename);
        formData.append("hash", hash);
        
        try
        {
            const response = await fetch("load_cached_audio", {
                method: "POST",
                body: formData
            });
    
            if (!response.ok) throw new Error('Upload failed.');
    
            const data = await response.json();
            dropArea.style.display = "none";
            globalAudioData = data;
            globalFile = [{
                name: data.filename[0], //! Added [0] because server now returns a list
                audio_url: data.audio_url[0] //! Same as above
            }];
            await visualizeAllFiles(globalFile); //! Changed arguments for new version of function
        }
        catch(err)
        {
            console.log(err);
            alert("Failed to load cached file.");
        }
        finally
        {
            spinner.style.display = 'none';
            isLoading = false;
        }
    }
    else
    {
        alert("Please wait for loading to finish");
    }
}


//! Added everything below
const drop_area = document.getElementById('drop_area');
const hidden_input = document.getElementById('hiddenFileInput');

// When user clicks the drop area, trigger the file dialog
drop_area.addEventListener('click',() =>
{
    hidden_input.click();
});

// When files are selected via the dialog, call handleDrop
hidden_input.addEventListener('change',(event) =>
{
    const files = event.target.files;

    // Create a fake event object with dataTransfer.files to reuse your handleDrop
    const fakeEvent = {
        preventDefault: () => {},
        dataTransfer: { files }
    };

    drop_area.classList.add('solid');
    handleDrop(fakeEvent);

    // Clear input so the same file can be selected again if needed
    hidden_input.value = '';
});