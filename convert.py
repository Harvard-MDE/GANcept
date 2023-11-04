import nbformat
from nbconvert import PythonExporter

def notebook_to_python(input_notebook, output_script):
    # Specify the encoding when opening the notebook file
    with open(input_notebook, 'r', encoding='utf-8') as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    exporter = PythonExporter()

    script, _ = exporter.from_notebook_node(notebook_content)

    with open(output_script, 'w', encoding='utf-8') as script_file:
        script_file.write(script)

if __name__ == "__main__":
    input_notebook = 'GAN_Dissection_notebook.ipynb'  # Replace with the path to your Jupyter Notebook
    output_script = 'GAN_Dissection_notebook.py'  # Replace with the desired output script file name

    notebook_to_python(input_notebook, output_script)


