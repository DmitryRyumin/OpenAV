const video = document.getElementById('video');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const uploadButton = document.getElementById('uploadButton');
const nextButton = document.getElementById('next');
let mediaRecorder;
let chunks = [];

const parts = [];
let flag = false;
let counter = 0;
let sec = 0;
let mins = 0;

// let t1 = "Question 1";
// let t2 = "Question 2";
// let t3 = "Question 3";
// let t4 = "Question 4";
// let t5 = "Question 5";
// let t6 = "Congratulations!";

// let q1 = "Introduce yourself, such as your name, age, specialty, study, and a few pieces of information that express you.";
// let q2 = "What skills do you have that will help you work in a potential company, where would you like to go to work?";
// let q3 = "Do you like to do sales?";
// let q4 = "Why do you want to do sales today?";
// let q5 = "What achievements are you proud of over the past 3 years?";
// let q6 = "You finished the questions, click on " + "<b>" + "Stop recording" + "</b>";


const constraints = {
    audio: {
        sampleRate: 48000,
        channelCount: 2,
        bitrate: 1536000,
    },
    video: {
        width: { min: 1280, ideal: 1920 },
        height: { min: 720, ideal: 1080 },
        frameRate: 30,
    }
    }

const timingData = [];
let nextButtonEnabled = true;

// Function to enable the "Next" button
function enableNextButton() {
    nextButtonEnabled = true;
    nextButton.disabled = false;
    var questionElement = document.getElementById('question');
    questionElement.style.visibility = "visible";
}

document.getElementById('next').onclick = function () {
    counter += 1;
    fetch('/get_questions')
        .then(response => response.json())
        .then(data => {
            const currentTime = new Date().toLocaleTimeString();
            var currentQuestion = data.questions.find(question => question.QuestionNumber === counter);
            if (currentQuestion) {
                var titleElement = document.getElementById('title');
                var questionElement = document.getElementById('question');
                titleElement.innerHTML = 'Вопрос ' + currentQuestion.QuestionNumber;
                questionElement.innerHTML = currentQuestion.QuestionText;
                timingData.push({ question: currentQuestion.QuestionNumber, timestamp: currentTime });
            } else {
                stop();
                return;
            }
            if (nextButtonEnabled) {
                // Disable the "Next" button
                nextButtonEnabled = false;
                nextButton.disabled = true;
                var questionElement = document.getElementById('question');
                questionElement.style.visibility = "hidden";
                console.log(currentQuestion.Disable_time);
                setTimeout(enableNextButton, currentQuestion.Disable_time);
            }
        })
        .catch(error => {
            console.error('An error occurred:', error);
        });
};


navigator.mediaDevices.getUserMedia(constraints)
    .then(function (stream) {
        video.srcObject = stream;

        // let sec = 0;
        //     let mins = 0;
            let timer = setInterval(function(){
                if (flag==true)
                {
                sec++;
                mins = Math.floor(sec/60);
                let mm = mins%60;
                let ss = sec%60;
                if (mm<10){ mm = "0" + mm;}
                if (ss<10){ ss = "0" + ss;}
                document.getElementById('Timer').innerHTML= mm + ":" + ss;
                }
            }, 1000);


        mediaRecorder = new MediaRecorder(stream, {mimeType: 'video/webm; codecs=vp9'});
        // mediaRecorder.start(1000);

        mediaRecorder.ondataavailable = function (event) {
            if (event.data.size > 0) {
                chunks.push(event.data);
            }
        };

        mediaRecorder.onstop = function () {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const videoUrl = URL.createObjectURL(blob);
            video.src = videoUrl;
            // uploadButton.disabled = false;
        };
    });

startButton.addEventListener('click', () => {
    flag = true;
    document.getElementById('image').src="../static/images/green.png";
    mediaRecorder.start();
    startButton.disabled = true;
    stopButton.disabled = false;
    nextButton.disabled = false;
    
    // Capture the current timestamp
    const currentTime = new Date().toLocaleTimeString();
    // Add the timing data to the array
    timingData.push({ question: 'Start_recording', timestamp: currentTime });
});

function stop() {
    flag = false;
    mins = 0;
    sec = 0;
    document.getElementById('image').src = "../static/images/red.png";
    mediaRecorder.stop();
    startButton.disabled = false;
    stopButton.disabled = true;
    nextButton.disabled = true;

    // After stopping the recording, upload the video
    mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }

        if (mediaRecorder.state === "inactive") {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const formData = new FormData();
            formData.append('video', blob);

            fetch('/upload', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.blob())
            .then(data => {
                const url = window.URL.createObjectURL(data);
                // If you want to download the uploaded video, uncomment the following lines
                // const a = document.createElement('a');
                // a.href = url;
                // a.download = 'uploaded_video.webm';
                // a.click();
                chunks = [];
                uploadButton.disabled = false;
            });
        }
    }
}

stopButton.addEventListener('click', stop);

const userNameInput = document.getElementById('userName'); // Get a reference to the input field

// Function to enable a button
function enableButton(button) {
    button.removeAttribute('disabled');
    button.classList.remove('disabled');
}

// Function to disable a button
function disableButton(button) {
    button.setAttribute('disabled', true);
    button.classList.add('disabled');
}

// Function to check if the user's name is entered and enable the "Start recording" button
function checkNameAndEnableButton() {
    if (userNameInput.value.trim() !== '') {
        enableButton(startButton);
    } else {
        disableButton(startButton);
    }
}

// Add an event listener to the input field to check the name and enable/disable the button
userNameInput.addEventListener('input', checkNameAndEnableButton);


// Function to get the user's name from the input field
function getUserName() {
    const userNameInput = document.getElementById('userName');
    return userNameInput.value.trim(); // Trim to remove any leading or trailing whitespace
}

function clearUserName() {
    const userNameInput = document.getElementById('userName');
    userNameInput.value = ''; // Set the input field value to an empty string
    startButton.disabled = true;
    uploadButton.disabled = true;
}

// Call the clearUserName function when the page is loaded
window.addEventListener('load', clearUserName);

uploadButton.addEventListener('click', () => {
    const userName = getUserName();
    
    // You can use the userName to name the downloaded video
    const videoFileName = `${userName}.webm`;
    const textFileName = `${userName}.txt`;


    // Send the timing data to the server
    fetch('/store_timing_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(timingData),
    })
        .then(response => response.text())
        .then(data => {
            console.log(data); // Log the response from the server
        });

    
    // After processing, initiate automatic downloads
    fetch('/download_processed_video')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = videoFileName;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
        });

    // fetch('/download_timing_data')
    //     .then(response => response.blob())
    //     .then(blob => {
    //         const url = window.URL.createObjectURL(blob);
    //         const a = document.createElement('a');
    //         a.href = url;
    //         a.download = textFileName;
    //         document.body.appendChild(a);
    //         a.click();
    //         window.URL.revokeObjectURL(url);
            
    //     });
        // Create a text representation of the timingData array
        const timingDataText = timingData.map(item => `Question: ${item.question}, Timestamp: ${item.timestamp}`).join('\n');

        // Create a Blob with the text data and trigger the download
        const blob = new Blob([timingDataText], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = textFileName; // Set the filename
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);

        // Reset the timingData array
        timingData.length = 0;


        // timingData.length = 0;
        counter = 0;
        let sec = 0;
        let mins = 0;

        var questionElement = document.getElementById('question');
        questionElement.style.visibility = "visible";

        var titleElement = document.getElementById('title');
        var questionElement = document.getElementById('question');
        titleElement.innerHTML = "Инструкции";
        questionElement.innerHTML = "Вопросы появятся здесь, когда вы нажмете кнопку Далее.";
        
        uploadButton.disabled = true;
        

});

// // Function to check the processing status
// function checkProcessingStatus() {
//     fetch('/check_processing_status')
//         .then(response => response.text())
//         .then(status => {
//             if (status === 'True') {
//                 // Processing is finished, enable the "Upload and Process" button
//                 uploadButton.disabled = false;
//             } else {
//                 // Processing is not finished, check again after a delay
//                 setTimeout(checkProcessingStatus, 1000); // Check every 1 second
//             }
//         });
// }

// // Call the checkProcessingStatus function to start checking
// checkProcessingStatus();


