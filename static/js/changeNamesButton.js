let isOriginal = true;

const originalNames = {
    'spectral_centroid': 'Spectral Centroid',
    'spectral_flux': 'Spectral Flux',
    'spectral_deviation': 'Spectral Standard Deviation',
    'zerocrossingrate': 'Zero Crossing Rate',
    'amplitude': 'Amplitude',
    'yin_f0_librosa': 'Yin F0',
    "normalized_height": '(Spectral Centroid - Deviation)/2', 
    'none': 'None'
};

const newNames = {
    'spectral_centroid': 'Center of Spectrum',
    'spectral_flux': 'Change in Spectrum',
    'spectral_deviation': 'Spectrum Variability',
    'zerocrossingrate': 'Noisiness',
    'amplitude': 'Volume Level',
    'yin_f0_librosa': 'Pitch Estimation (Yin)',
    "normalized_height": 'Normalized Spectral height',
    'none': 'No Feature'
};

// document.getElementById('changeNamesButton').addEventListener('click', function() {
//     var selects = document.querySelectorAll('.featureSelect');
//     var names = isOriginal ? newNames : originalNames;

//     // Loop through each select element
//     selects.forEach(function(select) {
//         var options = select.options;
//         // Loop through options and change the names
//         for (var i = 0; i < options.length; i++) {
//             var option = options[i];
//             if (names[option.value]) {
//                 option.text = names[option.value];
//             }
//         }
//     });

//     // Toggle the state
//     isOriginal = !isOriginal;
// });