// Function to update the date and time
function updateDateTime() {
    const now = new Date();
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'};
    const formattedDateTime = `As of ${now.toLocaleDateString('en-US', options)}`;
    document.getElementById('currentDateTime').textContent = formattedDateTime;
}

// Update the date and time every second
setInterval(updateDateTime, 1000);
updateDateTime(); // Initial call
