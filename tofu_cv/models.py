from django.db import models

class DashboardData(models.Model):
    chart_type = models.CharField(max_length=50)  # 'pie', 'bar', 'line' 등
    data = models.JSONField()  # JSON 형식으로 저장
    created_at = models.DateTimeField(auto_now_add=True)
