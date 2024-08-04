
from django.urls import path
from .views import menu, red_light_green_light, dalgona_game, quit_game, process_frame , start_dalgona_game , execute_redlight
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', menu, name='menu'),
    path('red_light_green_light/', red_light_green_light, name='red_light_green_light'),
    path('dalgona/', dalgona_game, name='dalgona'),
    path('quit/', quit_game, name='quit'),
    path('process_frame/', process_frame, name='process_frame'),
    path('start_dalgona/', start_dalgona_game, name='start_dalgona'),
    path('execute_redlight/', execute_redlight, name='execute_redlight'),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


