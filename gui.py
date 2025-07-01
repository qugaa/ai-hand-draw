"""
GUI launcher for the PDF Gesture Annotation Tool using FreeSimpleGUI.
Launch this script (`gui.py`) to open the launcher window.
"""
import FreeSimpleGUI as sg
import main  # Import main module functions (load_pdf and run_annotation_loop)

# --- Styling Configuration ---
sg.set_options(font=('Helvetica', 12))        # Use a consistent font for the GUI
sg.theme('LightGrey2')                       # Set a modern, minimal light-grey theme for the window

# --- GUI Layout Definition ---
layout = [
    [sg.Text('PDF Gesture Annotation Tool', font=('Helvetica', 20, 'bold'),
             justification='center', expand_x=True)],  # Title text (bold, centered)
    [sg.Text('Use hand gestures to draw, erase, and navigate PDF pages.',
             font=('Helvetica', 12), justification='center', expand_x=True)],  # Subtitle/instruction text
    [
        sg.Button('Open & Annotate PDF', size=(20, 2), font=('Helvetica', 14)),
        sg.Button('Quit', size=(10, 2), font=('Helvetica', 14),
                  button_color=('white', 'tomato'))  # Quit button with white text on a tomato-red background
    ]
]

# Create the Window
window = sg.Window(
    'PDF Gesture Annotator Launcher',  # Window title
    layout,
    element_justification='center',   # Center all elements in the window
    finalize=True                     # Finalize window creation (needed to manipulate window after creation if necessary)
)

# --- Event Loop for the Launcher ---
while True:
    event, _ = window.read()  # Read GUI events (no input values needed here)  # type: ignore[reportUndefinedVariable](falsepositive)
    if event in (sg.WIN_CLOSED, None, 'Quit'):
        # If window is closed or user clicks "Quit", exit the loop
        break
    if event == 'Open & Annotate PDF':
        # 1. Load the PDF (show file dialog and prepare pages)
        if main.load_pdf():            # Only proceed if a PDF was successfully loaded
            # 2. Run the main OpenCV gesture annotation loop
            main.run_annotation_loop()
        # 3. After the annotation window is closed, bring the launcher back to front
        window.bring_to_front()
        window.refresh()

window.close()  # Close the launcher window when exiting the loop
