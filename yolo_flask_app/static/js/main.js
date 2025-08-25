document.addEventListener('DOMContentLoaded', (event) => {
    const videoStream = document.getElementById('video-stream');

    // Make sure the video stream element exists before adding listeners
    if (videoStream) {
        // --- Existing code for handling clicks on the video ---
        videoStream.addEventListener('click', (e) => {
            const rect = videoStream.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const displayWidth = videoStream.clientWidth;
            const displayHeight = videoStream.clientHeight;

            const nativeWidth = videoStream.dataset.videoWidth;
            const nativeHeight = videoStream.dataset.videoHeight;

            const scaleX = nativeWidth / displayWidth;
            const scaleY = nativeHeight / displayHeight;

            const videoX = Math.round(clickX * scaleX);
            const videoY = Math.round(clickY * scaleY);

            fetch('/select_object', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ x: videoX, y: videoY }),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Server response from click:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });

        // --- New code for the reset button goes here ---
        const resetButton = document.getElementById('reset-button');

        // Also check if the reset button exists
        if (resetButton) {
            resetButton.addEventListener('click', () => {
                fetch('/reset_selection', { method: 'POST' }) // Ensure this matches your Flask route name
                .then(response => response.json())
                .then(data => {
                    console.log('Reset response:', data);
                    // The video stream will update automatically as the backend logic changes
                    // No need to reload the src if the backend state is handled correctly
                })
                .catch(error => console.error('Error:', error));
            });
        }
    }
});