"""Seed the OpenSearch index with the sample restaurant dataset."""

import sys
import os

# Ensure project root is on path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.document import Document
from app.services.semantic_search_service import index_documents

RESTAURANT_DATA = [
    {
        "id": "1",
        "title": "Restaurant Overview",
        "content": (
            "Welcome to DineSmart Restaurant, offering a wide variety of cuisines including "
            "Italian, Asian, and Continental dishes. Our chefs use fresh ingredients to create "
            "high-quality meals. The restaurant provides dine-in, takeaway, and delivery services. "
            "Open daily from 10 AM to 10 PM."
        ),
    },
    {
        "id": "2",
        "title": "Italian Menu",
        "content": (
            "Our Italian menu includes pasta, pizza, lasagna, and risotto. Popular dishes include "
            "Margherita Pizza, Alfredo Pasta, and Spaghetti Bolognese. We also offer gluten-free "
            "pasta and vegan pizza options."
        ),
    },
    {
        "id": "3",
        "title": "Asian Cuisine",
        "content": "We serve sushi, ramen, fried rice, and noodles. Fresh sushi and sashimi are prepared daily.",
    },
    {
        "id": "4",
        "title": "Desserts and Beverages",
        "content": "We offer coffee, juices, soft drinks, chocolate cake, ice cream, cheesecake, and brownies.",
    },
    {
        "id": "5",
        "title": "Vegan Options",
        "content": "We provide vegan meals, salads, smoothie bowls, and plant-based dishes.",
    },
    {
        "id": "6",
        "title": "Reservations",
        "content": "Customers can reserve tables online or by phone. Recommended during weekends.",
    },
    {
        "id": "7",
        "title": "Delivery",
        "content": "We deliver within 5 km radius. Orders can be placed online.",
    },
    {
        "id": "8",
        "title": "Opening Hours",
        "content": "Open daily from 10 AM to 10 PM. Peak hours are 7 PM to 9 PM.",
    },
]


def main():
    documents = [Document(**d) for d in RESTAURANT_DATA]
    count = index_documents(documents)
    print(f"Successfully indexed {count} chunks from {len(documents)} documents.")


if __name__ == "__main__":
    main()
