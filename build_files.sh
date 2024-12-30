 echo "BUILD START"
 python manage.py collectstatic --noinput --clear
 python manage.py migrate
 echo "BUILD END"
