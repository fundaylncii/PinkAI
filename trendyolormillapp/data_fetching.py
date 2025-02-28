
import requests
import re
import pandas as pd
import cloudscraper

def get_reviews(product_link, get_page = 100, save_path=False, channel_no = 0):
    content_id_match = re.search(r'p-(\d+)', product_link)
    if content_id_match:
        content_id = content_id_match.group(1)
        print(f"Content ID: {content_id}")
    else:
        print("Geçerli bir URL'de contentId bulunamadı.")
        return None

    if channel_no == 1:
        api_url = f"https://apigw.trendyol.com/discovery-web-websfxsocialreviewrating-santral/product-reviews-detailed?&contentId={content_id}"
    elif channel_no == 2:
        api_url = f"https://apigw.trendyol-milla.com/discovery-web-websfxsocialreviewrating-santral/product-reviews-detailed?&contentId={content_id}"
    else:
        print("Geçerli bir kanal id girini! (Trendyol : 1 , Trendyolmilla : 2)")
        return None
        

    print(api_url)
    scraper = cloudscraper.create_scraper()
    response = scraper.get(api_url)

    if response.status_code != 200:
        print(f"API çağrısı başarısız! HTTP Durum Kodu: {response.status_code}")
        return None

    try:
        json_data = response.json()
        product_reviews = json_data.get("result", {}).get("productReviews", {})

        total_pages = int(product_reviews.get("totalPages", 0))
        if total_pages == 0:
            print("Ürün için yorum bulunamadı.")
            return None

        print(f"Toplam Sayfa Sayısı: {total_pages}")

    except Exception as e:
        print(f"JSON verisi işlenirken hata oluştu: {e}")
        return None

    ## total_pages_input = input(f"Kaç sayfa veri çekmek istiyorsunuz? (Max: {total_pages}): ")
    total_pages_input = get_page
    try:
        total_pages = min(int(total_pages_input), total_pages)
    except ValueError:
        print("Geçersiz giriş! Tüm sayfalar çekilecek.")
    
    reviews_list = []

    for page in range(1, total_pages + 1):
        paginated_url = f"{api_url}&page={page}"
        response = scraper.get(paginated_url)

        if response.status_code != 200:
            print(f"Sayfa {page} alınamadı! HTTP Durum Kodu: {response.status_code}")
            continue

        try:
            json_data = response.json()
            print(f"Sayfa {page} için API yanıtı: {json_data}")  # API verisini kontrol et

            product_reviews = json_data.get("result", {}).get("productReviews", {})

            if not product_reviews:
                print(f"Sayfa {page} için 'productReviews' bulunamadı.")
                continue

            content = product_reviews.get("content", [])

            for review in content:
                reviews_list.append({
                    "reviewId": review.get("id"),
                    "rate": review.get("rate"),
                    "comment": review.get("comment"),
                    "commentDate": review.get("commentDateISOtype"),
                    "sellerName": review.get("sellerName"),
                    "productSize": review.get("productSize"),
                    "productHeight": review.get("productAttributes", {}).get("height"),
                    "productWeight": review.get("productAttributes", {}).get("weight"),
                    "reviewLikeCount": review.get("reviewLikeCount"),
                    "imageUrl": review.get("mediaFiles", [{}])[0].get("url") if review.get("mediaFiles") else None
                })
        except Exception as e:
            print(f"Sayfa {page} JSON hatası: {e}")
            continue

    if not reviews_list:
        print("Hiç yorum çekilemedi.")
        return None

    print("Veri çekme işlemi tamamlandı!")
    
    df = pd.DataFrame(reviews_list)

    if save_path:
        file_path = "reviews.xlsx"
        try:
            df.to_excel(file_path, index=False)
            print(f"Dosya '{file_path}' olarak başarıyla kaydedildi.")
        except Exception as e:
            print(f"Dosya kaydedilirken bir hata oluştu: {e}")

    return df
