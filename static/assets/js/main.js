function openModal() 
{   document.getElementById("myModal").style.display = "block";
}

function closeModal() 
{   document.getElementById("myModal").style.display = "none";
}

window.onclick = function(event) 
{   var modal = document.getElementById("myModal");
    if (event.target == modal)  
    {   modal.style.display = "none";
    }
}

function setButtonValue(value) 
{   document.getElementById("button_clicked").value = value;
    var fileInputsContainer = document.getElementById("fileInputsContainer");
    fileInputsContainer.innerHTML = "";
    for (var i = 0; i < parseInt(value.charAt(0)); i++) 
    {   var input = document.createElement("input");
        input.setAttribute("type", "file");
        input.setAttribute("name", "csv_file_" + (i + 1));
        input.setAttribute("id", "csv_file_" + (i + 1));
        input.setAttribute("accept", ".csv");

        // Add event listener to update label when file is selected
        input.addEventListener('change', function(e) 
        {   var file = e.target.files[0];
            var label = document.querySelector('label[for="' + e.target.id + '"]');
            if (file) 
            {   label.textContent = file.name;
                checkFilesSelected(); // Check if all files are selected
            } 
            else 
            {   if (parseInt(value.charAt(0)) === 1) 
                {   label.textContent = "Choose File";
                } 
                else 
                {   if (e.target.id === "csv_file_1") 
                    {   label.textContent = "Train Data Set";
                    } 
                    else 
                    {   label.textContent = "Test Data Set";
                    }
                }
                hideSubmitButton(); // Hide submit button if no file is selected
            }
        });
        fileInputsContainer.appendChild(input);
        var label = document.createElement("label");
        if (parseInt(value.charAt(0)) === 1) 
        {   label.textContent = "Choose File";
        } 
        else 
        {   label.textContent = (i === 0) ? "Train Data Set" : "Test Data Set";
        }
        label.setAttribute("for", "csv_file_" + (i + 1));
        fileInputsContainer.appendChild(label);
    }
}

function checkFilesSelected() 
{   var fileInputs = document.querySelectorAll('input[type="file"]');
    var allFilesSelected = true;
    fileInputs.forEach(function(input) 
    {   if (!input.files[0]) 
        {   allFilesSelected = false;
        }
    });
    if (allFilesSelected) 
    {   showSubmitButton();
    } 
    else 
    {   hideSubmitButton();
    }
}

function showSubmitButton() 
{   document.getElementById("submitBtn").style.display = "block";
}

function hideSubmitButton() 
{   document.getElementById("submitBtn").style.display = "none";
}

document.getElementById('scroll-btn').addEventListener('click', function() {
    var chatItem = document.querySelectorAll('.chatbox .chat');
    var lastchatItem = chatItem[chatItem.length - 1];
    lastchatItem.scrollIntoView();
});

function opengraph() 
{   document.getElementById("graphModal").style.display = "block";
    document.querySelector('.visualization').innerHTML = '';
}

function closegraph() 
{   document.getElementById("graphModal").style.display = "none";
}

document.addEventListener("DOMContentLoaded", function() {
    var scrollButton = document.getElementById("scroll-btn");

    window.addEventListener("scroll", function() {
        // Check if there is vertical scroll in the chatbox
        var chatbox = document.querySelector(".chatbox");
        if (chatbox.scrollHeight > chatbox.clientHeight) {
            // If there is vertical scroll, show the scroll button
            scrollButton.style.display = "block";
        } else {
            // If there is no vertical scroll, hide the scroll button
            scrollButton.style.display = "none";
        }
    });
});

function opencode() {
    document.getElementById('codeModal').style.display = "block";

    // Get the text from the pre tag
    let text = document.getElementById('dialog1Title').innerText;

    // Create a temporary textarea and append it to the body
    let textarea = document.createElement("textarea");
    document.body.appendChild(textarea);
  
    // Set the value of the textarea and select it
    textarea.value = text;
    textarea.select();
  
    // Copy the text and remove the textarea
    document.execCommand('copy');
    document.body.removeChild(textarea);
}

function closecode() {
    document.getElementById('codeModal').style.display = "none";
}

//step2.html

//step3.html
document.getElementById('next_step_3').addEventListener('click', function(event) {
    event.preventDefault();

    var form = document.getElementById('select_columns_form');
    var checkboxes = document.getElementById('user_preprocessing_steps').querySelectorAll('input[type=checkbox]');
    
    checkboxes.forEach(function(checkbox) {
        // when checkbox is checked add a hidden input to the form
        if (checkbox.checked) {
            var input = document.createElement('input');
            input.type = 'hidden';
            input.name = checkbox.id;
            input.value = '1';
            form.appendChild(input);
        }

        // disable the checkbox to prevent its value from being included in the form data
        checkbox.disabled = true;
    });

    if (checkboxes.length !== 0) {
        console.log("Form data before submitting:", [...new FormData(form)]);
        form.submit();
    } else {
        console.log("No checkboxes selected. Exiting...");
    }
});

function setYColumnState(graphGroup) {
    const graphTypeSelect = graphGroup.querySelector('#graph_type');
    const ySelect = graphGroup.querySelector('#y_column');
    const xSelect = graphGroup.querySelector('#x_column');

    function validateColumns() {
        if (graphTypeSelect.value !== "Pie" && graphTypeSelect.value !== "Box" && graphTypeSelect.value !== "Histogram") {
            if (xSelect.value === ySelect.value) {
                alert('The X Column and Y Column cannot be the same.');
            }
        }
    }

    graphTypeSelect.addEventListener('change', function() {
        if (this.value === "Pie" || this.value === "Box" || this.value === "Histogram") {
            ySelect.setAttribute('disabled', '');
        } else {
            ySelect.removeAttribute('disabled');
        }
        validateColumns();
    });

    xSelect.addEventListener('change', validateColumns);
    if(ySelect) {
        ySelect.addEventListener('change', validateColumns);
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    var newGraphBtn = document.getElementById("new_graph");
    if(newGraphBtn){
        newGraphBtn.addEventListener("click", function() {
            var graphGroup = document.getElementById('graph_group');
            if(graphGroup){
                var clone = graphGroup.cloneNode(true);
                clone.id = "graph_group" + Math.floor(Math.random() * 1000);

                setYColumnState(clone); // Setting up the onChange behavior for the new graph group

                // Enabling the y_column select box for the new group
                clone.querySelector('#y_column').removeAttribute('disabled');

                var graphChoice = document.querySelector(".graph-choice");
                if(graphChoice){
                    graphChoice.appendChild(clone);
                }
            }
        });
    }

    var nextButton = document.getElementById("new_next_btn");
    if(nextButton){
        nextButton.addEventListener("click", function() {
            console.log("Next button clicked");

            var dropdownGroups = document.querySelectorAll('div[id^="graph_group"]');
            var allGroupData = [];

            dropdownGroups.forEach((group, index) => {
                var dropdowns = group.querySelectorAll('.graph-column');
                var groupData = {};

                dropdowns.forEach((dropdown) => {
                    var selectedOption = dropdown.options[dropdown.selectedIndex].value;
                    groupData[dropdown.id] = selectedOption;
                });

                console.log("Group data #" + (index + 1) + ": ", groupData);
                allGroupData.push(groupData);
            });

            console.log("All group ", allGroupData);
            sendDataToServer(allGroupData);
        });
    }

    var defaultGraphGroup = document.getElementById('graph_group');
    if (defaultGraphGroup) setYColumnState(defaultGraphGroup); // Setting up the onChange behavior for the default graph group
});

function sendDataToServer(allData) {
    // Show the processing dialog
    const dialog = document.getElementById('processingDialog');
    dialog.showModal();

    fetch('/data_visualization', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ allDropdownData: allData })
    })
    .then(response => response.text())
    .then(html => {
        // Overwrite the page entirely
        document.open();
        document.write(html);
        document.close();
    })
    .catch((error) => {
        // In case of error overwrite the entire page with the error
        document.open();
        document.write("<p>An error occurred while processing your request. Please refresh the page and try again.</p>");
        document.close();

        console.error('Error:', error);
    });
}

//step4.html














