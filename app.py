import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from utils import (
    UPLOAD_FOLDER, REPORT_FOLDER, run_plagiarism_check,
    generate_pdf_report, read_file_content # Added read_file_content here if needed separately
)

# Ensure upload and report directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_very_secret_key_for_dev") # Use env var for production

ALLOWED_EXTENSIONS = {'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # Generate a unique filename to avoid collisions
        unique_id = uuid.uuid4().hex
        filename = f"{unique_id}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            flash(f'File "{original_filename}" uploaded successfully. Processing...')
            print(f"File saved to: {filepath}")

            # Run the check - This can take time!
            # Consider using background tasks (Celery, RQ) for real applications
            analysis_result = run_plagiarism_check(filepath)

            # Store results in session to pass to the results page & for PDF generation
            session['analysis_results'] = analysis_result
            session['original_filename'] = original_filename

            # Clean up the uploaded file immediately after processing
            try:
                os.remove(filepath)
                print(f"Cleaned up uploaded file: {filepath}")
            except OSError as e:
                print(f"Error deleting uploaded file {filepath}: {e}")

            if analysis_result.get("error"):
                flash(f"Error during processing: {analysis_result['error']}", 'error')
                return redirect(url_for('index')) # Redirect back if major error before results page

            return redirect(url_for('show_results'))

        except Exception as e:
            flash(f'An error occurred: {e}', 'error')
            print(f"Error during file save or processing trigger: {e}")
            # Clean up if file was saved but processing failed early
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError as del_e:
                    print(f"Error cleaning up partially processed file {filepath}: {del_e}")
            return redirect(request.url)

    else:
        flash('Invalid file type. Allowed types: txt, docx')
        return redirect(request.url)

@app.route('/results')
def show_results():
    results_data = session.get('analysis_results')
    original_filename = session.get('original_filename')

    if not results_data:
        flash('No analysis results found. Please upload a file first.', 'warning')
        return redirect(url_for('index'))

    # Clear session data after retrieving it? Optional, depends if you want refresh to work
    # session.pop('analysis_results', None)
    # session.pop('original_filename', None)

    return render_template('results.html',
                           results=results_data.get('results', []),
                           max_similarity=results_data.get('max_similarity', 0.0),
                           original_filename=original_filename,
                           error=results_data.get('error'))

@app.route('/download_report')
def download_report():
    results_data = session.get('analysis_results')
    original_filename = session.get('original_filename')

    if not results_data or not original_filename:
        flash('Cannot generate report. Analysis data missing.', 'error')
        return redirect(url_for('index'))

    if results_data.get("error"):
        flash(f"Cannot generate report due to previous error: {results_data['error']}", 'error')
        return redirect(url_for('show_results')) # Redirect to results page showing the error

    # Generate unique report filename
    report_id = uuid.uuid4().hex
    base_filename = os.path.splitext(original_filename)[0]
    pdf_filename = f"report_{base_filename}_{report_id}.pdf"
    pdf_filepath = os.path.join(app.config['REPORT_FOLDER'], pdf_filename)

    success = generate_pdf_report(
        original_filename=original_filename,
        results_data=results_data.get('results', []),
        output_filepath=pdf_filepath
    )

    if success and os.path.exists(pdf_filepath):
        try:
            # Send the file, then schedule deletion (or handle cleanup differently)
            # Note: send_from_directory is safer for serving files
            return send_from_directory(app.config['REPORT_FOLDER'], pdf_filename, as_attachment=True)
            # Cleanup could be done via a background task or periodically
        except Exception as e:
            print(f"Error sending PDF file: {e}")
            flash('Could not send the generated PDF report.', 'error')
            return redirect(url_for('show_results'))
    else:
        flash('Failed to generate the PDF report.', 'error')
        return redirect(url_for('show_results'))

# --- Serve Static Files (Optional, good for development) ---
# In production, use a proper web server (Nginx, Apache) to serve static files
# @app.route('/static/<path:filename>')
# def static_files(filename):
#     return send_from_directory('static', filename)

if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True)
