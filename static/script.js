const textArea = document.getElementById("promptText");
const submitButton = document.getElementById("submitButton");
const statusMessage = document.getElementById("statusMessage");
const results = document.getElementById("results");
const confidenceElement = document.getElementById("confidence");
const detectedItems = document.getElementById("detectedItems");

let checkTimer;

function setStatus(message, statusClass) {
    statusMessage.textContent = message;
    statusMessage.className = `status ${statusClass}`;
}

function clearResults() {
    results.hidden = true;
    confidenceElement.textContent = "0%";
    detectedItems.replaceChildren();
}

function showPatternMatches(matches) {
    detectedItems.replaceChildren();

    if (matches.length === 0) {
        const message = document.createElement("p");
        message.textContent =
            "The model detected possible PII, but no simple pattern was identified.";

        detectedItems.appendChild(message);
        return;
    }

    const heading = document.createElement("p");
    heading.textContent = "Detected information:";
    detectedItems.appendChild(heading);

    const list = document.createElement("ul");

    for (const match of matches) {
        const item = document.createElement("li");
        item.textContent = `${match.type}: ${match.value}`;
        list.appendChild(item);
    }

    detectedItems.appendChild(list);
}

async function checkForPII() {
    const text = textArea.value.trim();

    if (!text) {
        submitButton.disabled = true;
        setStatus("Enter text to begin checking.", "neutral");
        clearResults();
        return;
    }

    submitButton.disabled = true;
    setStatus("Checking text for sensitive information...", "checking");

    try {
        const response = await fetch("/api/check-pii", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        const result = await response.json();

        results.hidden = false;
        confidenceElement.textContent =
            `${Math.round(result.confidence * 100)}%`;

        if (result.contains_pii) {
            submitButton.disabled = true;

            setStatus(
                "Warning: possible personal information detected. " +
                "Review and remove it before submitting.",
                "danger"
            );

            showPatternMatches(result.pattern_matches);
        } else {
            submitButton.disabled = false;

            setStatus(
                "No likely personal information was detected.",
                "safe"
            );

            detectedItems.replaceChildren();
        }
    } catch (error) {
        console.error(error);

        submitButton.disabled = true;

        setStatus(
            "The PII checker could not be reached. Submission remains disabled.",
            "danger"
        );
    }
}

textArea.addEventListener("input", () => {
    clearTimeout(checkTimer);
    submitButton.disabled = true;
    checkTimer = setTimeout(checkForPII, 500);
});

submitButton.addEventListener("click", () => {
    alert(
        "Prompt approved by the prototype PII checker. " +
        "No prompt was sent to an actual AI service."
    );
});