from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Create initial users with specified credentials'

    def handle(self, *args, **options):
        User = get_user_model()

        credentials = [
            { 'name': 'Super Admin', 'email': 'superadmin@accusaga.ai','password': 'Admin@123', 'role': 'superadmin', 'is_approved': True}
        ]

        for credential in credentials:
            if not User.objects.filter(email=credential['email']).exists():
                user = User.objects.create_user(
                    email=credential['email'],
                    password=credential['password'],
                    role=credential['role'],
                    name=credential['name'],
                    is_approved=credential['is_approved']
                )
