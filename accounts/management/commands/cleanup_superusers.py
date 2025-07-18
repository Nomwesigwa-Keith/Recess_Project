from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from accounts.models import UserProfile

class Command(BaseCommand):
    help = 'Remove UserProfile entries for superusers to ensure they use Django admin only'

    def handle(self, *args, **options):
        superusers = User.objects.filter(is_superuser=True)
        removed_count = 0
        
        for superuser in superusers:
            try:
                profile = UserProfile.objects.get(user=superuser)
                profile.delete()
                removed_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Removed UserProfile for superuser: {superuser.username}')
                )
            except UserProfile.DoesNotExist:
                self.stdout.write(
                    self.style.WARNING(f'No UserProfile found for superuser: {superuser.username}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully removed {removed_count} UserProfile entries for superusers')
        ) 