 echo "BUILD START"
 python3.11.4 manage.py collectstatic --noinput --clear
 python3.11.4 manage.py migrate
 echo "BUILD END"
