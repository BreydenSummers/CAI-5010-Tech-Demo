import os
import json
import subprocess
import sys # To get the current python interpreter path
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Add Secret Key --- 
# Needed for session management (e.g., flash messages)
# IMPORTANT: Keep this key secret in production! Use environment variables.
app.secret_key = os.urandom(24) 
# ----------------------

# Configuration
UPLOAD_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure the results directory exists
os.makedirs(UPLOAD_ROOT, exist_ok=True)

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
                    model_path = os.path.join(workspace_root, 'extractor', 'model', 'test_model.pth') # Point to the specific model file

                    # Construct command
                    # Use arguments identified from script's usage string
                    cmd = [
                        sys.executable, 
                        script_path,
                        '--input', save_path,      # Changed from --image_path
                        '--model', model_path,      # Changed from --model_path
                        '--output_dir', project_path # Added output directory
                        # Add other necessary args for extract.py here
                        # e.g., '--confidence', '0.5'
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
                    # Catch other potential errors during subprocess execution
                    error_msg = f"An unexpected error occurred during analysis of {filename}: {str(e)}"
                    print(error_msg)
                    analysis_errors.append(error_msg)
                # --- End analysis script --- 
        
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
            
            # Get the base name without the _detection suffix
            base = fname.replace("_detection", "")
            metadata_path = os.path.join(project_path, f"{base}.json")
            
            # Load metadata if available
            metadata = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as mfile:
                        metadata = json.load(mfile)
                except json.JSONDecodeError:
                    metadata = None
            
            image_entries.append({
                "filename": base,  # Store the original filename without _detection
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
    # analysis_output = your_ai_analysis_function(project_path, selected_filenames)
    # For now, the "result" is just the list of filenames selected.
    analysis_output = {
        "message": "Placeholder analysis complete.",
        "analyzed_files": selected_filenames,
        "details": "Further details like trends, stats, etc., would go here."
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
        
        # Get the base name without the _detection suffix
        base = fname.replace("_detection", "")
        metadata_path = os.path.join(project_path, f"{base}.json")
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as mfile:
                    metadata = json.load(mfile)
            except json.JSONDecodeError:
                metadata = None
        
        image_entries.append({
            "filename": base,  # Store the original filename without _detection
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

# (In a real setup, you would include app.run or use a WSGI server to run the app)
