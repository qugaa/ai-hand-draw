import FreeSimpleGUI as sg
import main

# Styling Configuration
sg.set_options(font=('Helvetica', 12))
sg.theme('LightGrey2')

# GUI Layout Definition
layout = [
    [sg.Text('PDF Gesture Annotation Tool', font=('Helvetica', 20, 'bold'),
             justification='center', expand_x=True)],
    [sg.Text('Use hand gestures to draw, erase, and navigate PDF pages.',
             font=('Helvetica', 12), justification='center', expand_x=True)],
    [
        sg.Button('Open & Annotate PDF', size=(20, 2), font=('Helvetica', 14)),
        sg.Button('Quit', size=(10, 2), font=('Helvetica', 14),
                  button_color=('white', 'tomato'))
    ]
]

window = sg.Window(
    'PDF Gesture Annotator Launcher',
    layout,
    element_justification='center',
    finalize=True
)

# Event Loop for the Launcher
while True:
    event, _ = window.read()  # type: ignore[reportUndefinedVariable](falsepositive)
    if event in (sg.WIN_CLOSED, None, 'Quit'):
        break
    if event == 'Open & Annotate PDF':
        if main.load_pdf():
            main.run_annotation_loop()
        window.bring_to_front()
        window.refresh()

window.close()
