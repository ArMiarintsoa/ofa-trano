"""
Scraper pour MadaRent - récupère les annonces de location.
Utilise Playwright pour exécuter le JS, puis BeautifulSoup pour parser.
"""

import time
import random
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re

# --- Configuration ---
BASE_URL = "https://app.madarent.mg"
OUTPUT_DIR = Path("data/raw")
DELAY_RANGE = (2, 5)  # secondes entre chaque requête, pour être respectueux du serveur
NBR_CARDS_PER_PAGE = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_rendered_html(page, url: str) -> str:
    """Charge une page et attend que le JS ait fini de rendre le contenu."""
    page.goto(url, wait_until="networkidle", timeout=30000)
    # Attente explicite supplémentaire si le site charge en plusieurs vagues
    page.wait_for_timeout(1500)
    return page.content()

async def fetch_rendered_html(url: str) -> str:
    with sync_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)  # laisse le temps au JS de tout charger
        html = page.content()
        browser.close()
    return html

def parse_address(card) -> dict:
    """Extrait ville et quartier depuis .address-row en se basant sur
    l'icône, pas sur la position (plus robuste si l'ordre change)."""
    address = {"ville": None, "lieu": None}
    
    address_row = card.select_one(".address-row")
    if not address_row:
        return address
    
    for bullet in address_row.select("span.bullet"):
        icon = bullet.select_one("iconify-icon")
        icon_name = icon.get("icon") if icon else None
        
        # Le texte est après le tag iconify-icon, pas dedans
        text = bullet.get_text(strip=True)
        
        if icon_name == "solar:city-linear":
            address["ville"] = text
        elif icon_name == "solar:map-point-linear":
            address["lieu"] = text
    
    return address

def parse_essential_info(content) -> dict:
    info = {"pieces": None,"chambres": None, "cuisine": None, "toilettes": None, "Accès moto": None, "Accès voiture": None, "Charges": None}
    
    grid  = content.select_one("div.infos-grid")
    if not grid:
        return info
    bullets = grid.select("span.bullet")
    for bullet in bullets:                
        icon = bullet.select_one("iconify-icon")        
        icon_name = icon.get("icon") if icon else None
        text = bullet.get_text(strip=True)
        match icon_name:
            case "solar:home-2-linear":
                info["pieces"] = text
            case "solar:bed-linear":
                info["chambres"] = text
            case "solar:chef-hat-linear":
                info["cuisine"] = text
            case "solar:bath-linear":
                info["toilettes"] = text
            case "ph:motorcycle":
                info["Accès moto"] = text
            case "solar:lightbulb-linear":
                info["Charges"] = text
            case "ph:car":
                info["Accès voiture"] = text            
    return info

def parse_one_listing(html: str) -> dict:
    listing = {}    
    soup  = BeautifulSoup(html, "html.parser")    
    content = soup.select_one('.content')  
    address = parse_address(content)
    essential_info = parse_essential_info(content)
    listing = {**address,**essential_info,"description": _safe_text(content.select_one("#house-description")),"prix": _clean_price(_safe_text(content.select_one(".price-row span.price"))), "unité du prix":_safe_text(content.select_one(".price-row .price-unit")),"titre": _safe_text(content.select_one(".listing-title")), "durée": _safe_text(content.select_one(".meta-time span")), "note": _safe_text(content.select_one('.meta-rating span')), "vues": _safe_text(content.select_one('.meta-rating span.views')), "type": _safe_text(content.select_one('div.house-badges span.badge-ghost'))}    
    return listing

def parse_listings(html: str, page) -> list[dict]:
    """Parse le HTML rendu et extrait les annonces. 
    À AJUSTER selon les vrais sélecteurs CSS du site."""
    soup = BeautifulSoup(html, "html.parser")
    listings = []

    # PLACEHOLDER — remplace par le vrai sélecteur des cartes d'annonces
    cards = soup.select("div.house")  # ex: à identifier via l'inspecteur    

    for card in cards:
        try:
            link = card.select_one("a")["href"] if card.select_one("a") else None
            if link:
                try:
                    content = get_rendered_html(page,f'{BASE_URL}{link}')                                             
                except Exception as e:
                    logger.warning(f"Erreur parsing d'une annonce: {e}")
                    continue
                listing = parse_one_listing(content)                            
            
            listings.append(listing)
        except Exception as e:
            logger.warning(f"Erreur parsing d'une annonce: {e}")
            continue

    return listings

def _clean_price(text: str | None) -> int | None:
    """Convertit '300\u202f000 Ar' en 300000 (int)."""
    if not text:
        return None
    # Supprime tous les espaces (normaux ET insécables) et le texte "Ar"
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None

def _safe_text(element) -> str | None:
    """Évite les crashs si un élément n'existe pas dans une annonce donnée."""
    return element.get_text(strip=True) if element else None

def get_number_in_text(element: str) -> int:    
    # On extrait uniquement les chiffres, en ignorant le texte autour
    cleaned_text = element.get_text().replace(" ", "")
    match = re.search(r"\d+", cleaned_text)        
    return int(match.group()) if match else 0

def parse_number_results(html:str)-> int:
    soup = BeautifulSoup(html, "html.parser")
    # 1. On cible directement l'élément de manière propre
    element = soup.select_one("span.results-count")
    # 2. On extrait le nombre de manière robuste (Regex)
    result_count = 0
    if element:
        result_count = get_number_in_text(element)    

    return result_count


def calculateMaxPages(page) -> int:
    url = f"{BASE_URL}/recherche?sort=rating&page={1}" 
    try:
        html = get_rendered_html(page, url)
        import math
        return math.ceil(parse_number_results(html) / NBR_CARDS_PER_PAGE)

    except Exception as e:
        logger.error(f"Échec chargement page {1}: {e}")
    return 0
       


def scrape_all_pages() -> pd.DataFrame:
    all_listings = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()        

        MAX_PAGES = calculateMaxPages(page) 
        logger.info(f'Nombre de page: {MAX_PAGES}')

        for page_num in range(1, MAX_PAGES + 1):
            url = f"{BASE_URL}/recherche?sort=rating&page={page_num}"            
            logger.info(f"Scraping page {page_num}: {url}")            

            try:
                html = get_rendered_html(page, url)
            except Exception as e:
                logger.error(f"Échec chargement page {page_num}: {e}")
                continue

            listings = parse_listings(html, page)            

            if not listings:
                logger.info(f"Aucune annonce trouvée page {page_num}, arrêt.")
                break

            all_listings.extend(listings)
            logger.info(f"  -> {len(listings)} annonces récupérées")

            # Délai aléatoire pour ne pas surcharger le serveur
            time.sleep(random.uniform(*DELAY_RANGE))

        browser.close()

    return pd.DataFrame(all_listings)


def save_data(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filepath = OUTPUT_DIR / f"madarent_{timestamp}.csv"
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Données sauvegardées: {filepath} ({len(df)} lignes)")
    return filepath


def main():
    logger.info("Début du scraping MadaRent")
    df = scrape_all_pages()

    if df.empty:
        logger.warning("Aucune donnée récupérée — vérifier si le site a changé de structure.")
        return

    save_data(df)
    logger.info("Scraping terminé.")


if __name__ == "__main__":
    main()