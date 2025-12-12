/**
 * custom_select.js
 * Transforms native <select> elements into custom styled dropdowns.
 * Usage: Call setupCustomSelects() after DOM is ready.
 */

function setupCustomSelects(selector = 'select') {
    const selects = document.querySelectorAll(selector);

    selects.forEach(select => {
        // Skip if already processed or if it's explicitly excluded
        if (select.classList.contains('no-custom') || select.dataset.customized) return;
        select.dataset.customized = 'true';

        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'custom-select-wrapper';

        // Insert wrapper before select
        select.parentNode.insertBefore(wrapper, select);

        // Move select inside and hide it
        wrapper.appendChild(select);
        select.classList.add('original-select');

        // Get initial selected option
        const selectedOption = select.options[select.selectedIndex];
        const selectedText = selectedOption ? selectedOption.text : 'Select...';

        // Create trigger
        const trigger = document.createElement('div');
        trigger.className = 'custom-select-trigger';
        trigger.innerHTML = `
            <span class="selected-text">${selectedText}</span>
            <span class="arrow"></span>
        `;
        wrapper.appendChild(trigger);

        // Create options container
        const optionsContainer = document.createElement('div');
        optionsContainer.className = 'custom-options';

        Array.from(select.options).forEach((opt, idx) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'custom-option';
            optionDiv.dataset.value = opt.value;
            optionDiv.textContent = opt.text;

            if (idx === select.selectedIndex) {
                optionDiv.classList.add('selected');
            }

            optionDiv.addEventListener('click', (e) => {
                e.stopPropagation();

                // Update original select
                select.value = opt.value;

                // Trigger change event on the original select
                select.dispatchEvent(new Event('change', { bubbles: true }));

                // Update UI
                trigger.querySelector('.selected-text').textContent = opt.text;
                optionsContainer.querySelectorAll('.custom-option').forEach(o => o.classList.remove('selected'));
                optionDiv.classList.add('selected');

                // Close dropdown
                wrapper.classList.remove('open');
            });

            optionsContainer.appendChild(optionDiv);
        });

        wrapper.appendChild(optionsContainer);

        // Toggle open
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();

            // Close other open dropdowns
            document.querySelectorAll('.custom-select-wrapper.open').forEach(w => {
                if (w !== wrapper) w.classList.remove('open');
            });

            wrapper.classList.toggle('open');
        });
    });

    // Close on click outside
    document.addEventListener('click', () => {
        document.querySelectorAll('.custom-select-wrapper.open').forEach(w => {
            w.classList.remove('open');
        });
    });
}

/**
 * Refresh a single custom select if its options have changed dynamically.
 * @param {HTMLSelectElement} select - The original select element
 */
function refreshCustomSelect(select) {
    const wrapper = select.closest('.custom-select-wrapper');
    if (!wrapper) return;

    const optionsContainer = wrapper.querySelector('.custom-options');
    const trigger = wrapper.querySelector('.custom-select-trigger .selected-text');

    // Clear existing options
    optionsContainer.innerHTML = '';

    // Re-populate
    Array.from(select.options).forEach((opt, idx) => {
        const optionDiv = document.createElement('div');
        optionDiv.className = 'custom-option';
        optionDiv.dataset.value = opt.value;
        optionDiv.textContent = opt.text;

        if (idx === select.selectedIndex) {
            optionDiv.classList.add('selected');
            trigger.textContent = opt.text;
        }

        optionDiv.addEventListener('click', (e) => {
            e.stopPropagation();
            select.value = opt.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            trigger.textContent = opt.text;
            optionsContainer.querySelectorAll('.custom-option').forEach(o => o.classList.remove('selected'));
            optionDiv.classList.add('selected');
            wrapper.classList.remove('open');
        });

        optionsContainer.appendChild(optionDiv);
    });
}
