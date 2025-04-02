import os
import random
import re
import requests
import docx # python-docx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher # Good for finding exact phrase matches
from urllib.parse import quote_plus # For URL encoding search phrases
from fpdf import FPDF # For PDF generation

# --- Configuration ---
# !! IMPORTANT: Replace with your actual ScrapingBee API Key !!
# If you don't have one, the google_search function will use placeholders.
SCRAPINGBEE_API_KEY = os.environ.get("SCRAPINGBEE_API_KEY", "42DKOJ2K4OBVKILTVBCZ32ETDJFDAM0OM4QTWQSH2F50E1IG1FFYWD5JZQS4038W1PBF3EO6P89XPSJI")
# Make sure SCRAPINGBEE_API_KEY is set as an environment variable or replace the placeholder
# Example: export SCRAPINGBEE_API_KEY='your_real_key' (in Linux/macOS)
# Example: set SCRAPINGBEE_API_KEY=your_real_key (in Windows cmd)

NUM_PHRASES = 5       # How many phrases to extract and search
PHRASE_MIN_WORDS = 7
PHRASE_MAX_WORDS = 12
NUM_SEARCH_RESULTS = 5 # Top N results per phrase
SIMILARITY_THRESHOLD = 0.1 # Minimum similarity score to report a URL
MATCH_BLOCK_THRESHOLD = 6 # Minimum word count for a "matching block"

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# --- File Handling ---

def read_file_content(filepath):
    """Reads content from supported file types."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith('.docx'):
        try:
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs if para.text])
        except Exception as e:
            print(f"Error reading DOCX {filepath}: {e}")
            raise ValueError(f"Could not read DOCX file: {e}")
    elif filepath.endswith('.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT {filepath}: {e}")
            raise ValueError(f"Could not read TXT file: {e}")
    else:
        raise ValueError("Unsupported file format. Please upload .docx or .txt")

# --- Text Processing ---

def extract_phrases(text, num_phrases=NUM_PHRASES, min_words=PHRASE_MIN_WORDS, max_words=PHRASE_MAX_WORDS):
    """Extracts random phrases from the text."""
    # Basic cleaning: replace multiple whitespaces/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    if len(words) < min_words:
        return [text] if text else [] # Return the whole text if it's short

    extracted = set() # Use a set to avoid duplicate phrases initially
    attempts = 0
    max_attempts = num_phrases * 10 # Limit attempts to prevent infinite loops

    while len(extracted) < num_phrases and attempts < max_attempts:
        attempts += 1
        if len(words) <= max_words: # Handle cases where text is shorter than max_words
            start_index = 0
            phrase_len = len(words)
        else:
            phrase_len = random.randint(min_words, max_words)
            if len(words) - phrase_len <= 0: continue # Should not happen due to outer check, but safeguard
            start_index = random.randint(0, len(words) - phrase_len)

        phrase = " ".join(words[start_index : start_index + phrase_len])

        # Simple check to avoid phrases ending mid-sentence weirdly (optional)
        # if phrase[-1] in ['.', '?', '!']:
        extracted.add(phrase)

    return list(extracted)


# --- Web Scraping ---

def google_search(phrase, api_key=SCRAPINGBEE_API_KEY, num_results=NUM_SEARCH_RESULTS):
    """Performs Google search using ScrapingBee (or simulates if no key)."""
    search_query = f'"{phrase}"' # Use quotes for exact phrase match
    print(f"Searching for: {search_query}")

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("WARN: SCRAPINGBEE_API_KEY not set. Returning placeholder URLs.")
        # Simulate results for testing without API key
        return [f"https://example.com/placeholder?search={quote_plus(phrase)}&n={i+1}" for i in range(num_results)]

    try:
        api_url = 'https://app.scrapingbee.com/api/v1/'
        params = {
            'api_key': api_key,
            'url': f'https://www.google.com/search?q={quote_plus(search_query)}&num={num_results}',
            'render_js': 'false', # Usually not needed for Google search results text
            # Add other params like 'country_code' if needed
        }
        response = requests.get(api_url, params=params, timeout=60) # Increase timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- Parsing ScrapingBee Google Search Results ---
        # This depends heavily on ScrapingBee's output format.
        # Let's assume it returns JSON with a list of organic results.
        # Adjust parsing based on actual API response structure.
        # Example structure assumption: {'organic_results': [{'link': '...', 'title': '...'}]}

        data = response.json()
        urls = [result.get('link') for result in data.get('organic_results', []) if result.get('link')]

        # Fallback or alternative: Parse HTML if JSON structure is different or unavailable
        if not urls and response.ok: # Check if status code was 200
            soup = BeautifulSoup(response.text, 'lxml') # Use lxml for speed
            # Google's structure changes! This is fragile. Inspect actual results.
            # Common pattern: Links within <h3> tags, often inside specific divs.
            links = []
            for h3 in soup.find_all('h3'):
                a_tag = h3.find('a')
                if a_tag and a_tag.get('href'):
                    href = a_tag.get('href')
                    # Clean up Google's redirect URLs if necessary
                    if href.startswith('/url?q='):
                        href = href.split('/url?q=')[1].split('&sa=')[0]
                    if href.startswith('http') and 'google.com' not in href: # Avoid internal google links
                        links.append(href)
            urls = links

        print(f"Found {len(urls)} URLs for '{phrase}'.")
        return urls[:num_results]

    except requests.exceptions.RequestException as e:
        print(f"Error during Google search via ScrapingBee for '{phrase}': {e}")
        return []
    except Exception as e:
        print(f"Error processing search results for '{phrase}': {e}")
        return []


def scrape_and_clean(url):
    """Scrapes the main text content from a given URL."""
    print(f"Scraping: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20) # Timeout for fetching page
        response.raise_for_status()

        # Check content type - only parse HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Skipping non-HTML content: {url} ({content_type})")
            return None

        soup = BeautifulSoup(response.content, 'lxml') # Use response.content for encoding detection

        # 1. Remove common non-content tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'input']):
            tag.decompose()

        # 2. Attempt to find main content area (common tags/attributes)
        # This is heuristic and might need tuning for different website structures
        main_content = (
                soup.find('article') or
                soup.find('main') or
                soup.find('div', id=re.compile(r'main|content|entry|post', re.I)) or
                soup.find('div', class_=re.compile(r'main|content|post|body|article', re.I))
        )

        # 3. If specific main area found, extract text primarily from <p> tags within it
        if main_content:
            text_elements = main_content.find_all('p')
            # If few <p> tags, get broader text from main_content
            if len(text_elements) < 3 :
                text = main_content.get_text(" ", strip=True)
            else:
                text = " ".join(p.get_text(" ", strip=True) for p in text_elements)
        else:
            # Fallback: Extract text from all <p> tags in the body if no main area identified
            paragraphs = soup.find_all('p')
            if paragraphs:
                text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
            else:
                # Final fallback: get all text from body, trying to preserve structure slightly better
                body = soup.find('body')
                text = body.get_text(" ", strip=True) if body else ""


        # 4. Final cleaning - normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        print(f"Scraped {len(cleaned_text)} chars from {url}")
        return cleaned_text

    except requests.exceptions.Timeout:
        print(f"Timeout error scraping {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error scraping {url}: {e}")
        return None
    except Exception as e:
        print(f"General error scraping or cleaning {url}: {e}")
        return None # Return None on any scraping error

# --- Comparison ---

def calculate_overall_similarity(text1, text2):
    """Calculates similarity using TF-IDF and Cosine Similarity."""
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except ValueError as e:
        # Can happen if text is too short or contains only stop words after processing
        print(f"TF-IDF Error: {e}. Texts might be too short or lack meaningful content.")
        return 0.0 # Return 0 similarity if TF-IDF fails

def find_matching_phrases(original_text, scraped_text, min_match_len=MATCH_BLOCK_THRESHOLD):
    """Finds specific matching text blocks using SequenceMatcher."""
    if not original_text or not scraped_text:
        return []

    # Using split() is simple but loses punctuation context.
    # Consider more advanced tokenization if needed.
    original_words = original_text.lower().split()
    scraped_words = scraped_text.lower().split()

    matcher = SequenceMatcher(None, original_words, scraped_words, autojunk=False)

    matching_phrases = []
    for match in matcher.get_matching_blocks():
        # match attributes: a (start in A), b (start in B), size (length)
        if match.size >= min_match_len:
            # Extract the matched segment from the *original* text
            start_index = match.a
            end_index = match.a + match.size
            matched_phrase = " ".join(original_words[start_index:end_index])
            matching_phrases.append(matched_phrase)

    # Return unique phrases, maybe sorted by length or occurrence? For now, just unique.
    return sorted(list(set(matching_phrases)), key=len, reverse=True) # Longer matches first

# --- PDF Generation (Extra Challenge) ---

def generate_pdf_report(original_filename, results_data, output_filepath):
    """Generates a PDF report of the plagiarism check results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Plagiarism Check Report", ln=1, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Original File: {original_filename}", ln=1)
    pdf.ln(10)

    reported_sources = 0
    for result in results_data:
        # Only include sources above the similarity threshold in the report
        if result['similarity'] >= SIMILARITY_THRESHOLD:
            reported_sources += 1
            pdf.set_font("Arial", 'B', 11)
            # Use multi_cell for long URLs that need wrapping
            pdf.multi_cell(0, 6, f"Source URL ({result['similarity']:.2%}): {result['url']}", border=0) # Changed border to 0
            pdf.ln(2)

            if result['matching_phrases']:
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 6, "Matching Phrases/Sections Found:", ln=1)
                pdf.set_font("Arial", 'I', 9)
                for phrase in result['matching_phrases'][:5]: # Limit phrases shown per source in PDF
                    # Encode potentially problematic characters for Latin-1 (FPDF default)
                    safe_phrase = phrase.encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 5, f"- {safe_phrase}", border=0) # Changed border to 0
                if len(result['matching_phrases']) > 5:
                    pdf.cell(0, 5, "- ... (and more)", ln=1)
                pdf.ln(3)
            else:
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 6, "No specific long matching phrases found (check overall similarity).", ln=1)
                pdf.ln(3)

            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y()) # Draw a separator line
            pdf.ln(5)

    if reported_sources == 0:
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "No significant matching content found based on the search.", ln=1)

    try:
        pdf.output(output_filepath)
        print(f"PDF report generated: {output_filepath}")
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

# --- Main Processing Function ---

def run_plagiarism_check(filepath):
    """Orchestrates the entire plagiarism check process."""
    results = []
    try:
        original_content = read_file_content(filepath)
        if not original_content or len(original_content) < PHRASE_MIN_WORDS * 5: # Basic check for content length
            print("Document content seems too short for effective checking.")
            return {"error": "Document content too short or unreadable.", "results": []}

        search_phrases = extract_phrases(original_content)
        if not search_phrases:
            print("Could not extract phrases to search.")
            return {"error": "Could not extract searchable phrases from the document.", "results": []}

        print(f"Extracted Phrases: {search_phrases}")

        unique_urls_to_scrape = set()
        for phrase in search_phrases:
            urls = google_search(phrase)
            for url in urls:
                unique_urls_to_scrape.add(url)

        print(f"\nUnique URLs to scrape: {len(unique_urls_to_scrape)}")

        scraped_data = {} # Store scraped content to avoid re-scraping same URL
        for url in unique_urls_to_scrape:
            if url not in scraped_data:
                scraped_text = scrape_and_clean(url)
                scraped_data[url] = scraped_text # Store even if None, to mark attempt

        print("\nComparing content...")
        final_results = []
        for url, scraped_content in scraped_data.items():
            if scraped_content:
                similarity_score = calculate_overall_similarity(original_content, scraped_content)

                # Find specific matches only if overall similarity is somewhat significant
                matching_phrases_list = []
                if similarity_score > (SIMILARITY_THRESHOLD / 2): # Lower threshold for trying seq matcher
                    matching_phrases_list = find_matching_phrases(original_content, scraped_content)

                # Only store results above the defined threshold
                if similarity_score >= SIMILARITY_THRESHOLD or matching_phrases_list:
                    final_results.append({
                        'url': url,
                        'similarity': similarity_score,
                        'matching_phrases': matching_phrases_list,
                        'scraped_preview': scraped_content[:200] + "..." # Add a preview
                    })
            else:
                print(f"Skipping comparison for {url} due to scraping failure.")


        # Sort results by similarity score, highest first
        final_results.sort(key=lambda x: x['similarity'], reverse=True)

        # Calculate an overall score (e.g., max similarity found)
        max_similarity = max(r['similarity'] for r in final_results) if final_results else 0.0

        return {
            "max_similarity": max_similarity,
            "results": final_results,
            "error": None
        }

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"error": str(e), "results": []}
    except ValueError as e: # Catches unsupported format or read errors from read_file_content
        print(f"Error: {e}")
        return {"error": str(e), "results": []}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return {"error": f"An unexpected server error occurred: {e}", "results": []}
    finally:
        # Clean up the uploaded file? Optional, depends on strategy.
        # If storing permanently, don't delete. If temporary, delete here.
        # try:
        #     if os.path.exists(filepath): os.remove(filepath)
        # except OSError as e:
        #     print(f"Error deleting uploaded file {filepath}: {e}")
        pass
