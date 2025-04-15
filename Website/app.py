import os
import json
import subprocess
import sys # To get the current python interpreter path
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, flash
from werkzeug.utils import secure_filename
from openai import OpenAI
import base64

app = Flask(__name__)

# --- Add Secret Key ---
# Needed for session management (e.g., flash messages)
# IMPORTANT: Keep this key secret in production! Use environment variables.
app.secret_key = os.urandom(24)
# ----------------------

# Add zip function to Jinja2 environment
app.jinja_env.globals.update(zip=zip)

# Configuration
UPLOAD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure the results directory exists
os.makedirs(UPLOAD_ROOT, exist_ok=True)

def encode_image_base64(image_path):
    """Encode an image to base64."""
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

def allowed_file(filename):
    """Check if a filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Homepage: list all projects and form to create a new project."""
    projects_info = []
    # List all subdirectories in results/ as projects
    for name in sorted(os.listdir(UPLOAD_ROOT)):
        proj_path = os.path.join(UPLOAD_ROOT, name)
        if os.path.isdir(proj_path):
            # Count original images (ignore annotated variants)
            images = [f for f in os.listdir(proj_path)
                      if allowed_file(f) and "_annotated" not in f]
            projects_info.append({"name": name, "count": len(images)})
    return render_template("index.html", projects=projects_info)

@app.route("/create_project", methods=["POST"])
def create_project():
    """Handle creation of a new project (folder)."""
    project_name = request.form.get("project_name", "").strip()
    if project_name == "":
        # If no name provided, redirect back (could add flash message for real app)
        return redirect(url_for("index"))
    # Sanitize the project name to a safe folder name
    safe_name = secure_filename(project_name)
    if safe_name == "":
        return redirect(url_for("index"))
    project_path = os.path.join(UPLOAD_ROOT, safe_name)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    # Redirect to the new project's page
    return redirect(url_for("project_view", project_name=safe_name))

@app.route("/project/<project_name>", methods=["GET", "POST"])
def project_view(project_name):
    """Project page: upload images, trigger analysis, and display gallery."""
    safe_proj = secure_filename(project_name)
    project_path = os.path.join(UPLOAD_ROOT, safe_proj)

    print(f"Project view for: {project_name}")
    print(f"Project path: {project_path}")

    if not os.path.isdir(project_path):
        print(f"Project directory not found: {project_path}")
        abort(404)

    if request.method == "POST":
        # Handle file upload
        files = request.files.getlist("images")
        if not files or files[0].filename == "":
            flash("No files selected", "error")
            return redirect(url_for("project_view", project_name=project_name))

        analysis_errors = []
        analysis_success = 0

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(project_path, filename)
                file.save(save_path)
                print(f"Saved uploaded file to: {save_path}")

                # --- Trigger analysis script ---
                try:
                    # Define paths relative to workspace root
                    workspace_root = os.path.abspath(os.path.join(app.root_path, '..'))
                    script_path = os.path.join(workspace_root, 'extractor', 'extract.py')
                    model_path = os.path.join(workspace_root, 'extractor', 'model', 'test_model.pth')

                    # Construct command
                    cmd = [
                        sys.executable,
                        script_path,
                        '--input', save_path,
                        '--model', model_path,
                        '--output_dir', project_path,
                        '--confidence', '0.5'
                    ]
                    print(f"Running analysis command: {' '.join(cmd)}")

                    # Run the script from the workspace root directory
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=workspace_root)
                    print(f"Analysis successful for {filename}:\n{result.stdout}")
                    analysis_success += 1

                except subprocess.CalledProcessError as e:
                    error_msg = f"Error analyzing {filename}: {e.stderr}"
                    print(error_msg)
                    analysis_errors.append(error_msg)
                except FileNotFoundError:
                    error_msg = f"Error analyzing {filename}: Could not find analysis script or Python interpreter."
                    print(error_msg)
                    analysis_errors.append(error_msg)
                except Exception as e:
                    error_msg = f"An unexpected error occurred during analysis of {filename}: {str(e)}"
                    print(error_msg)
                    analysis_errors.append(error_msg)

        # Flash messages summarizing analysis results
        if analysis_success > 0:
            flash(f"Successfully processed and initiated analysis for {analysis_success} image(s).", "success")
        for error in analysis_errors:
            flash(error, "error")

        return redirect(url_for("project_view", project_name=project_name))

    else: # GET request
        # Gather images and any available annotations/metadata for display
        image_entries = []
        files = sorted(os.listdir(project_path))
        print(f"Files in project directory: {files}")

        # Only include files that have detection images
        for fname in files:
            # Skip non-image files and files that don't have detection images
            if not allowed_file(fname) or not fname.endswith(("_detection.png", "_detection.jpg", "_detection.jpeg")):
                continue

            # Get the base name without the _detection suffix and extension
            base_with_ext = fname.replace("_detection", "")
            base_name, ext = os.path.splitext(base_with_ext)

            # Try different possible JSON file paths
            possible_json_paths = [
                os.path.join(project_path, f"{base_name}.json"),  # Original format
                os.path.join(project_path, f"{base_name}_detection.json"),  # New format
                os.path.join(project_path, f"{base_with_ext}.json"),  # With extension
                os.path.join(project_path, f"{base_with_ext}_detection.json")  # With extension
            ]

            # Try each possible path
            metadata = None
            for metadata_path in possible_json_paths:
                print(f"Looking for metadata at: {metadata_path}")
                print(f"File exists: {os.path.exists(metadata_path)}")

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as mfile:
                            metadata = json.load(mfile)
                            # print(f"Loaded metadata for {base_name} from {metadata_path}: {metadata}")
                            break  # Found and loaded metadata, no need to try other paths
                    except Exception as e:
                        print(f"Error loading metadata from {metadata_path}: {e}")

            if metadata is None:
                print(f"No metadata found for {base_name} in any of the expected locations")

            # Debug print for metadata structure
            # if metadata:
            #     print(f"Metadata structure for {base_name}:")
            #     print(f"  Keys: {list(metadata.keys())}")
            #     if 'predictions' in metadata:
            #         print(f"  Predictions keys: {list(metadata['predictions'].keys())}")
            #         if 'scores' in metadata['predictions']:
            #             print(f"  Number of scores: {len(metadata['predictions']['scores'])}")
            #             print(f"  First few scores: {metadata['predictions']['scores'][:3] if len(metadata['predictions']['scores']) > 0 else 'None'}")

            image_entries.append({
                "filename": base_name,  # Store the original filename without extension
                "annotated": fname,  # Store the detection image filename
                "meta": metadata
            })

        print(f"Image entries to display: {[img['filename'] for img in image_entries]}")
        return render_template("project.html", project_name=project_name, images=image_entries)

@app.route("/project/<project_name>/analyze", methods=["POST"])
def analyze_selected(project_name):
    """Handle request to run further analysis on selected images and display results on the same page."""
    safe_proj = secure_filename(project_name)
    project_path = os.path.join(UPLOAD_ROOT, safe_proj)
    if not os.path.isdir(project_path):
        abort(404) # Project not found

    selected_filenames = request.form.getlist("selected_images")

    if not selected_filenames:
        flash("No images were selected for analysis.", "error")
        return redirect(url_for("project_view", project_name=project_name))

    print(f"Project '{project_name}': Received request to analyze {len(selected_filenames)} images:")
    print(selected_filenames)

    # --- Placeholder for AI analysis step ---
    if selected_filenames:
        first_image_path = f"{os.path.join(project_path, selected_filenames[0])}.jpg"
        try:
            client = OpenAI()

            # Encode the image to base64
            base64_image = encode_image_base64(first_image_path)

            # NOTE: Removed Metadata analysis for now. Kept getting weird results.

            response = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "system",
                        "content": "You are an expert imagery intelligence analyst. Your task is to analyze satellite images of airports and deliver your analysis in a structured JSON response."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            },
                        ]
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "airport_image_analysis",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "airport_type": {
                                    "type": "string",
                                    "description": "The classification of the airport.",
                                    "enum": [
                                        "Civil",
                                        "Military",
                                        "Joint"
                                    ]
                                },
                                "airport_type_evidence": {
                                    "type": "string",
                                    "description": "A brief explanation of the evidence supporting the airport type classifcation."
                                },
                                "operational_tempo": {
                                    "type": "string",
                                    # "description": "The current operational tempo of the airport. Consider the concentration and movement of planes, people, and vehicles.",
                                    "description": "The current operational tempo of the airport.",
                                    "enum": [
                                        "Low",
                                        "Moderate",
                                        "High",
                                        "Dormant"
                                    ]
                                },
                                "operational_tempo_reasoning": {
                                    "type": "string",
                                    "description": "A brief explanation of the reasoning supporting the operational tempo classification."
                                },
                                "abnormal_activity": {
                                    "type": "string",
                                    "description": 'Either a short description of any abnormal activity, or "None" if none is identified.',
                                },
                                "feature_summary": {
                                    "type": "string",
                                    "description": 'A brief summary of the image including notable features of the airport. Do not preface with "In this image" or "The image".',
                                }
                            },
                            "required": [
                                "airport_type",
                                "airport_type_evidence",
                                "operational_tempo",
                                "operational_tempo_reasoning",
                                "abnormal_activity",
                                "feature_summary",
                            ],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            # Extract response content
            gpt_analysis = response.output_text

            # Decode the JSON response
            try:
                gpt_analysis = json.loads(gpt_analysis)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response: {e}")
                flash("Error decoding analysis response.", "error")
                return redirect(url_for("project_view", project_name=project_name))
            print(f"Decoded JSON response: {gpt_analysis}")

            analysis_output = {
                "message": "Analysis completed.",
                "analyzed_files": selected_filenames,
                "details": gpt_analysis
            }

        except Exception as e:
            print(f"Error during API call: {e}")
            flash("Error during analysis.", "error")
            analysis_output = {
                "message": "Error during analysis.",
                "analyzed_files": selected_filenames,
                "details": str(e)
            }
    else:
        analysis_output = {
            "message": "No images selected for analysis.",
            "analyzed_files": [],
            "details": "No analysis performed."
        }
    # ----------------------------------------

    # We need to reload the image entries to re-render the project page correctly
    image_entries = []
    files = sorted(os.listdir(project_path))

    # Only include files that have detection images
    for fname in files:
        # Skip non-image files and files that don't have detection images
        if not allowed_file(fname) or not fname.endswith(("_detection.png", "_detection.jpg", "_detection.jpeg")):
            continue

        # Get the base name without the _detection suffix and extension
        base_with_ext = fname.replace("_detection", "")
        base_name, ext = os.path.splitext(base_with_ext)

        # Try different possible JSON file paths
        possible_json_paths = [
            os.path.join(project_path, f"{base_name}.json"),  # Original format
            os.path.join(project_path, f"{base_name}_detection.json"),  # New format
            os.path.join(project_path, f"{base_with_ext}.json"),  # With extension
            os.path.join(project_path, f"{base_with_ext}_detection.json")  # With extension
        ]

        # Try each possible path
        metadata = None
        # for metadata_path in possible_json_paths:
        #     print(f"Looking for metadata at: {metadata_path}")
        #     print(f"File exists: {os.path.exists(metadata_path)}")

        #     if os.path.exists(metadata_path):
        #         try:
        #             with open(metadata_path, "r") as mfile:
        #                 metadata = json.load(mfile)
        #                 print(f"Loaded metadata for {base_name} from {metadata_path}: {metadata}")
        #                 break  # Found and loaded metadata, no need to try other paths
        #         except Exception as e:
        #             print(f"Error loading metadata from {metadata_path}: {e}")

        # if metadata is None:
        #     print(f"No metadata found for {base_name} in any of the expected locations")

        # Debug print for metadata structure
        # if metadata:
        #     print(f"Metadata structure for {base_name}:")
        #     print(f"  Keys: {list(metadata.keys())}")
        #     if 'predictions' in metadata:
        #         print(f"  Predictions keys: {list(metadata['predictions'].keys())}")
        #         if 'scores' in metadata['predictions']:
        #             print(f"  Number of scores: {len(metadata['predictions']['scores'])}")
        #             print(f"  First few scores: {metadata['predictions']['scores'][:3] if len(metadata['predictions']['scores']) > 0 else 'None'}")

        image_entries.append({
            "filename": base_name,  # Store the original filename without extension
            "annotated": fname,  # Store the detection image filename
            "meta": metadata
        })

    # Re-render the project page, passing the analysis results
    return render_template(
        "project.html",
        project_name=project_name,
        images=image_entries, # Pass the full image list back
        analysis_results=analysis_output # Pass the new analysis results
    )

@app.route("/project/<project_name>/file/<filename>")
def uploaded_file(project_name, filename):
    """Serve uploaded files."""
    safe_proj = secure_filename(project_name)
    safe_file = secure_filename(filename)
    return send_from_directory(os.path.join(UPLOAD_ROOT, safe_proj), safe_file)

@app.route("/about")
def about():
    """Renders the about page."""
    return render_template("about.html")

@app.route("/project/<project_name>/delete/<filename>", methods=["POST"])
def delete_image(project_name, filename):
    """Delete an image and its associated files from a project."""
    safe_proj = secure_filename(project_name)
    safe_file = secure_filename(filename)
    project_path = os.path.join(UPLOAD_ROOT, safe_proj)

    print(f"Attempting to delete file: {safe_file} from project: {safe_proj}")
    print(f"Project path: {project_path}")

    if not os.path.isdir(project_path):
        print(f"Project directory not found: {project_path}")
        abort(404)

    # Get the base name and extension
    base, ext = os.path.splitext(safe_file)

    # List of files to delete
    files_to_delete = [
        safe_file,  # Original image
        f"{base}_detection{ext}",  # Detection version
        f"{base}.json"  # Metadata file
    ]

    print(f"Files to delete: {files_to_delete}")

    # Delete each file if it exists
    deleted_files = []
    for file in files_to_delete:
        file_path = os.path.join(project_path, file)
        print(f"Checking if file exists: {file_path}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Successfully deleted: {file_path}")
                deleted_files.append(file)
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
                flash(f"Error deleting {file}: {str(e)}", "error")
                return redirect(url_for("project_view", project_name=project_name))

    if deleted_files:
        flash(f"Successfully deleted {', '.join(deleted_files)}", "success")
    else:
        flash(f"No files were deleted. File {safe_file} may not exist.", "warning")

    print(f"Redirecting to project view: {project_name}")
    # Use a direct redirect to avoid any potential form submission
    return redirect(url_for("project_view", project_name=project_name), code=303)

@app.route('/project/<project_name>/image/<filename>')
def image_details(project_name, filename):
    """Display detailed information about a specific image."""
    project_dir = os.path.join(UPLOAD_ROOT, project_name)
    if not os.path.exists(project_dir):
        flash('Project not found.', 'error')
        return redirect(url_for('index'))

    # Get the detection image filename
    base_name = os.path.splitext(filename)[0]
    detection_filename = f"{base_name}_detection{os.path.splitext(filename)[1]}"

    # Load metadata from JSON file
    metadata = None
    json_path = os.path.join(project_dir, f"{base_name}_detection.json")
    print(f"Looking for metadata at: {json_path}")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"Loaded JSON data: {data}")
                if 'predictions' in data:
                    predictions = data['predictions']
                    metadata = {
                        'num_detections': len(predictions['scores']),
                        'avg_confidence': sum(predictions['scores']) / len(predictions['scores']) * 100,
                        'max_confidence': max(predictions['scores']) * 100,
                        'boxes': predictions['boxes'],
                        'scores': predictions['scores'],
                        'labels': predictions['labels']
                    }
                    print(f"Processed metadata: {metadata}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            flash('Error loading image metadata.', 'error')
    else:
        print(f"Metadata file not found at: {json_path}")
        # Try alternative paths
        alt_paths = [
            os.path.join(project_dir, f"{base_name}.json"),
            os.path.join(project_dir, "results", f"{base_name}_detection.json"),
            os.path.join(project_dir, "results", f"{base_name}.json")
        ]
        for alt_path in alt_paths:
            print(f"Trying alternative path: {alt_path}")
            if os.path.exists(alt_path):
                try:
                    with open(alt_path, 'r') as f:
                        data = json.load(f)
                        print(f"Loaded JSON data from alternative path: {data}")
                        if 'predictions' in data:
                            predictions = data['predictions']
                            metadata = {
                                'num_detections': len(predictions['scores']),
                                'avg_confidence': sum(predictions['scores']) / len(predictions['scores']) * 100,
                                'max_confidence': max(predictions['scores']) * 100,
                                'boxes': predictions['boxes'],
                                'scores': predictions['scores'],
                                'labels': predictions['labels']
                            }
                            print(f"Processed metadata from alternative path: {metadata}")
                            break
                except Exception as e:
                    print(f"Error loading metadata from alternative path: {e}")

    # Check if the original image exists
    original_path = os.path.join(project_dir, filename)
    if not os.path.exists(original_path):
        # Try with extension
        for ext in ['.jpg', '.jpeg', '.png']:
            alt_path = os.path.join(project_dir, f"{base_name}{ext}")
            if os.path.exists(alt_path):
                filename = f"{base_name}{ext}"
                break

    # Check if the detection image exists
    detection_path = os.path.join(project_dir, detection_filename)
    if not os.path.exists(detection_path):
        # Try with extension
        for ext in ['.jpg', '.jpeg', '.png']:
            alt_path = os.path.join(project_dir, f"{base_name}_detection{ext}")
            if os.path.exists(alt_path):
                detection_filename = f"{base_name}_detection{ext}"
                break

    print(f"Using original image: {filename}")
    print(f"Using detection image: {detection_filename}")

    # Ensure metadata has the required fields
    if metadata:
        if 'labels' not in metadata:
            metadata['labels'] = ['Object'] * len(metadata.get('scores', []))
        if 'scores' not in metadata:
            metadata['scores'] = []
        if 'boxes' not in metadata:
            metadata['boxes'] = []
        if 'num_detections' not in metadata:
            metadata['num_detections'] = len(metadata.get('scores', []))
        if 'avg_confidence' not in metadata:
            metadata['avg_confidence'] = sum(metadata.get('scores', [])) / len(metadata.get('scores', [])) * 100 if metadata.get('scores', []) else 0
        if 'max_confidence' not in metadata:
            metadata['max_confidence'] = max(metadata.get('scores', [])) * 100 if metadata.get('scores', []) else 0

    return render_template('image_details.html',
                         project_name=project_name,
                         filename=filename,
                         detection_filename=detection_filename,
                         metadata=metadata)

@app.route('/project/<project_name>/delete', methods=['POST'])
def delete_project(project_name):
    """Delete an entire project and all its contents."""
    project_dir = os.path.join(UPLOAD_ROOT, project_name)

    if not os.path.exists(project_dir):
        flash('Project not found.', 'error')
        return redirect(url_for('index'))

    try:
        # Remove all files in the project directory
        for filename in os.listdir(project_dir):
            file_path = os.path.join(project_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Remove the project directory itself
        os.rmdir(project_dir)
        flash('Project deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error deleting project: {str(e)}', 'error')

    return redirect(url_for('index'))

# (In a real setup, you would include app.run or use a WSGI server to run the app)
