function map(value, start1, stop1, start2, stop2) {
    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
}

// Function to generate a random value of -1 or 1
function getRandomSign() {
            return Math.random() < 0.5 ? -1 : 1;
}

