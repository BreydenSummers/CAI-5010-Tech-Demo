import os
import json
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Add Secret Key --- 
# Needed for session management (e.g., flash messages)
# IMPORTANT: Keep this key secret in production! Use environment variables.
app.secret_key = os.urandom(24) 
# ----------------------

# Configuration
UPLOAD_ROOT = os.path.join(os.getcwd(), "results")
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
    """Project page: upload images and display gallery for the given project."""
    project_path = os.path.join(UPLOAD_ROOT, secure_filename(project_name))
    # If project folder doesn't exist, return 404
    if not os.path.isdir(project_path):
        abort(404)
    if request.method == "POST":
        # Handle image upload(s)
        files = request.files.getlist("images")  # 'images' is the name of the file input
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(project_path, filename)
                file.save(save_path)
                # Placeholder: integrate analysis step here (e.g., generate annotated image and metadata)
                # For example:
                # annotated_path = os.path.join(project_path, f"{os.path.splitext(filename)[0]}_annotated{os.path.splitext(filename)[1]}")
                # result = analyze_image(save_path)  # (Pseudo-code) Analyze image, get annotations and metadata
                # result.save_annotated_image(annotated_path)
                # with open(os.path.join(project_path, f"{os.path.splitext(filename)[0]}.json"), "w") as meta_file:
                #     json.dump(result.metadata, meta_file)
        return redirect(url_for("project_view", project_name=project_name))
    else:
        # Gather images and any available annotations/metadata for display
        image_entries = []
        files = sorted(os.listdir(project_path))
        for fname in files:
            if not allowed_file(fname) or fname.endswith(("_annotated.png", "_annotated.jpg", "_annotated.jpeg")):
                continue  # skip non-images and annotated images in the main loop
            base, ext = os.path.splitext(fname)
            annotated_name = f"{base}_annotated{ext}"
            annotated_path = os.path.join(project_path, annotated_name)
            metadata_path = os.path.join(project_path, f"{base}.json")
            # Check if annotated version exists
            annotated = annotated_name if os.path.exists(annotated_path) else None
            # Load metadata if available
            metadata = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as mfile:
                        metadata = json.load(mfile)
                except json.JSONDecodeError:
                    metadata = None
            image_entries.append({
                "filename": fname,
                "annotated": annotated,
                "meta": metadata
            })
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
    for fname in files:
        if not allowed_file(fname) or fname.endswith(("_annotated.png", "_annotated.jpg", "_annotated.jpeg")):
            continue
        base, ext = os.path.splitext(fname)
        annotated_name = f"{base}_annotated{ext}"
        annotated_path = os.path.join(project_path, annotated_name)
        metadata_path = os.path.join(project_path, f"{base}.json")
        annotated = annotated_name if os.path.exists(annotated_path) else None
        metadata = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as mfile:
                    metadata = json.load(mfile)
            except json.JSONDecodeError:
                metadata = None
        image_entries.append({
            "filename": fname,
            "annotated": annotated,
            "meta": metadata
        })

    # Re-render the project page, passing the analysis results
    return render_template(
        "project.html", 
        project_name=project_name, 
        images=image_entries, # Pass the full image list back
        analysis_results=analysis_output # Pass the new analysis results
    )

@app.route("/results/<project_name>/<filename>")
def uploaded_file(project_name, filename):
    """Serve uploaded files (original or annotated images) from the results folder."""
    # Secure the paths to prevent path traversal
    safe_proj = secure_filename(project_name)
    safe_file = secure_filename(filename)
    dir_path = os.path.join(UPLOAD_ROOT, safe_proj)
    file_path = os.path.join(dir_path, safe_file)
    # Only send the file if it exists within the project directory
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(dir_path, safe_file)

@app.route("/about")
def about():
    """Renders the about page."""
    return render_template("about.html")

# (In a real setup, you would include app.run or use a WSGI server to run the app)
