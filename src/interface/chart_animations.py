"""
Graphiques animés pour HelixOne
Barres, lignes, donuts, sparklines avec animations
"""

import customtkinter as ctk
from tkinter import Canvas
import math
from typing import List, Optional

class AnimatedBarChart(Canvas):
    """Graphique en barres animé"""
    
    def __init__(self, parent, width: int = 600, height: int = 400, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg='#0f172a',
            highlightthickness=0,
            **kwargs
        )
        
        self.chart_width = width - 100
        self.chart_height = height - 100
        self.padding = 50
        
        self.labels = []
        self.values = []
        self.colors = []
        self.current_heights = []
    
    def set_data(self, labels: List[str], values: List[float], colors: List[str]):
        """Définit les données du graphique"""
        self.labels = labels
        self.values = values
        self.colors = colors
        self.current_heights = [0] * len(values)
        
        self._animate()
    
    def _animate(self):
        """Anime les barres"""
        all_done = True
        
        for i in range(len(self.values)):
            if self.current_heights[i] < self.values[i]:
                all_done = False
                self.current_heights[i] += (self.values[i] - self.current_heights[i]) * 0.1
        
        self._draw()
        
        if not all_done:
            self.after(30, self._animate)
    
    def _draw(self):
        """Dessine le graphique"""
        self.delete('all')
        
        if not self.values:
            return
        
        max_value = max(self.values) if self.values else 100
        bar_width = self.chart_width / len(self.values) * 0.8
        spacing = self.chart_width / len(self.values) * 0.2
        
        for i, (label, height) in enumerate(zip(self.labels, self.current_heights)):
            # Position
            x = self.padding + (i * (bar_width + spacing))
            bar_height = (height / max_value) * self.chart_height
            y = self.padding + self.chart_height - bar_height
            
            # Barre
            color = self.colors[i % len(self.colors)] if self.colors else '#3b82f6'
            self.create_rectangle(
                x, y,
                x + bar_width, self.padding + self.chart_height,
                fill=color,
                outline='',
                tags='bar'
            )
            
            # Label
            self.create_text(
                x + bar_width / 2,
                self.padding + self.chart_height + 20,
                text=label,
                fill='#8b949e',
                font=('Helvetica', 10),
                tags='label'
            )
            
            # Valeur
            self.create_text(
                x + bar_width / 2,
                y - 10,
                text=f"{height:.1f}",
                fill='#f0f6fc',
                font=('Helvetica', 11, 'bold'),
                tags='value'
            )
        
        # Axes
        self.create_line(
            self.padding, self.padding,
            self.padding, self.padding + self.chart_height,
            fill='#30363d',
            width=2,
            tags='axis'
        )
        
        self.create_line(
            self.padding, self.padding + self.chart_height,
            self.padding + self.chart_width, self.padding + self.chart_height,
            fill='#30363d',
            width=2,
            tags='axis'
        )


class AnimatedLineChart(Canvas):
    """Graphique en ligne animé"""
    
    def __init__(self, parent, width: int = 600, height: int = 300, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg='#0f172a',
            highlightthickness=0,
            **kwargs
        )
        
        self.chart_width = width - 100
        self.chart_height = height - 80
        self.padding = 50
        
        self.labels = []
        self.values = []
        self.color = '#3b82f6'
        self.progress = 0
    
    def set_data(self, labels: List[str], values: List[float], color: str = '#3b82f6'):
        """Définit les données du graphique"""
        self.labels = labels
        self.values = values
        self.color = color
        self.progress = 0
        
        self._animate()
    
    def _animate(self):
        """Anime la ligne"""
        if self.progress >= 1:
            self.progress = 1
            self._draw()
            return
        
        self.progress += 0.05
        self._draw()
        
        self.after(30, self._animate)
    
    def _draw(self):
        """Dessine le graphique"""
        self.delete('all')
        
        if not self.values or len(self.values) < 2:
            return
        
        # Calculer les points
        max_value = max(self.values)
        min_value = min(self.values)
        value_range = max_value - min_value if max_value > min_value else 1
        
        points = []
        visible_count = int(len(self.values) * self.progress)
        
        for i in range(visible_count):
            x = self.padding + (i / (len(self.values) - 1)) * self.chart_width
            normalized = (self.values[i] - min_value) / value_range
            y = self.padding + self.chart_height - (normalized * self.chart_height)
            points.append((x, y))
        
        # Dessiner la ligne
        if len(points) >= 2:
            line_points = []
            for x, y in points:
                line_points.extend([x, y])
            
            self.create_line(
                line_points,
                fill=self.color,
                width=3,
                smooth=True,
                tags='line'
            )
            
            # Points
            for x, y in points:
                self.create_oval(
                    x - 4, y - 4,
                    x + 4, y + 4,
                    fill=self.color,
                    outline='#ffffff',
                    width=2,
                    tags='point'
                )
        
        # Labels X
        for i, label in enumerate(self.labels):
            x = self.padding + (i / (len(self.labels) - 1)) * self.chart_width
            self.create_text(
                x, self.padding + self.chart_height + 20,
                text=label,
                fill='#8b949e',
                font=('Helvetica', 9),
                tags='label'
            )
        
        # Axes
        self.create_line(
            self.padding, self.padding,
            self.padding, self.padding + self.chart_height,
            fill='#30363d',
            width=2
        )
        
        self.create_line(
            self.padding, self.padding + self.chart_height,
            self.padding + self.chart_width, self.padding + self.chart_height,
            fill='#30363d',
            width=2
        )


class ProgressBar(Canvas):
    """Barre de progression animée"""
    
    def __init__(self, parent, width: int = 400, height: int = 30, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg='#0f172a',
            highlightthickness=0,
            **kwargs
        )
        
        self.bar_width = width - 20
        self.bar_height = height - 10
        self.progress = 0
        self.target_progress = 0
        
        self._draw_background()
    
    def _draw_background(self):
        """Dessine le fond de la barre"""
        self.create_rectangle(
            10, 5,
            10 + self.bar_width, 5 + self.bar_height,
            fill='#1e2329',
            outline='#30363d',
            width=1,
            tags='bg'
        )
    
    def set_progress(self, value: float):
        """Définit la progression (0-100)"""
        self.target_progress = max(0, min(100, value))
        self._animate()
    
    def _animate(self):
        """Anime la progression"""
        if abs(self.progress - self.target_progress) < 1:
            self.progress = self.target_progress
            self._draw()
            return
        
        self.progress += (self.target_progress - self.progress) * 0.1
        self._draw()
        
        self.after(30, self._animate)
    
    def _draw(self):
        """Dessine la barre de progression"""
        self.delete('progress')
        
        # Couleur selon la progression
        if self.progress >= 70:
            color = '#00ff88'
        elif self.progress >= 40:
            color = '#3b82f6'
        else:
            color = '#ff3860'
        
        # Barre de progression
        fill_width = (self.progress / 100) * self.bar_width
        self.create_rectangle(
            10, 5,
            10 + fill_width, 5 + self.bar_height,
            fill=color,
            outline='',
            tags='progress'
        )
        
        # Texte
        self.create_text(
            10 + self.bar_width / 2,
            5 + self.bar_height / 2,
            text=f"{int(self.progress)}%",
            fill='#f0f6fc',
            font=('Helvetica', 11, 'bold'),
            tags='progress'
        )


class SparklineChart(Canvas):
    """Mini graphique sparkline"""
    
    def __init__(self, parent, width: int = 150, height: int = 50, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg='transparent',
            highlightthickness=0,
            **kwargs
        )
        
        self.chart_width = width - 10
        self.chart_height = height - 10
        self.data = []
        self.color = '#3b82f6'
    
    def set_data(self, data: List[float], color: str = '#3b82f6'):
        """Définit les données"""
        self.data = data
        self.color = color
        self._draw()
    
    def _draw(self):
        """Dessine le sparkline"""
        self.delete('all')
        
        if len(self.data) < 2:
            return
        
        # Trouver min/max
        max_val = max(self.data)
        min_val = min(self.data)
        value_range = max_val - min_val if max_val > min_val else 1
        
        # Calculer les points
        points = []
        for i, value in enumerate(self.data):
            x = (i / (len(self.data) - 1)) * self.chart_width
            y = self.chart_height - ((value - min_val) / value_range * self.chart_height)
            points.append((x, y))
        
        # Dessiner la ligne
        if len(points) >= 2:
            line_points = []
            for x, y in points:
                line_points.extend([x, y])
            
            self.create_line(
                line_points,
                fill=self.color,
                width=2,
                smooth=True,
                tags='sparkline'
            )
            
            # Zone sous la courbe
            area_points = [0, self.chart_height]
            area_points.extend(line_points)
            area_points.extend([self.chart_width, self.chart_height])
            
            self.create_polygon(
                area_points,
                fill=self.color,
                outline='',
                stipple='gray50',
                tags='sparkline'
            )
            
            # Point final
            last_x, last_y = points[-1]
            self.create_oval(
                last_x - 3, last_y - 3,
                last_x + 3, last_y + 3,
                fill=self.color,
                outline='#ffffff',
                width=2,
                tags='sparkline'
            )


class DonutChart(Canvas):
    """Graphique en donut (anneau) animé"""
    
    def __init__(self, parent, size: int = 200, **kwargs):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg='transparent',
            highlightthickness=0,
            **kwargs
        )
        
        self.size = size
        self.center_x = size // 2
        self.center_y = size // 2
        self.outer_radius = (size - 40) // 2
        self.inner_radius = self.outer_radius - 30
        
        self.segments = []
        self.current_angle = 0
        self.target_angle = 0
    
    def set_data(self, values: List[float], labels: List[str], colors: List[str]):
        """Définit les données du donut"""
        self.segments = []
        total = sum(values)
        
        if total == 0:
            return
        
        start_angle = 90
        
        for i, value in enumerate(values):
            percentage = value / total
            extent = percentage * 360
            
            self.segments.append({
                'start': start_angle,
                'extent': extent,
                'color': colors[i % len(colors)],
                'label': labels[i] if i < len(labels) else f"Item {i+1}",
                'value': value,
                'percentage': percentage
            })
            
            start_angle += extent
        
        self.target_angle = 360
        self.current_angle = 0
        self._animate_donut()
    
    def _animate_donut(self):
        """Anime le donut"""
        if self.current_angle >= self.target_angle - 1:
            self.current_angle = self.target_angle
            self._draw_donut()
            return
        
        self.current_angle += (self.target_angle - self.current_angle) * 0.1
        self._draw_donut()
        
        self.after(16, self._animate_donut)
    
    def _draw_donut(self):
        """Dessine le donut"""
        self.delete('donut')
        self.delete('labels')
        
        drawn_angle = 0
        
        for segment in self.segments:
            if drawn_angle >= self.current_angle:
                break
            
            segment_angle = min(segment['extent'], self.current_angle - drawn_angle)
            
            # Arc externe
            self.create_arc(
                self.center_x - self.outer_radius,
                self.center_y - self.outer_radius,
                self.center_x + self.outer_radius,
                self.center_y + self.outer_radius,
                start=segment['start'],
                extent=segment_angle,
                fill=segment['color'],
                outline='',
                tags='donut'
            )
            
            # Arc interne (trou)
            self.create_arc(
                self.center_x - self.inner_radius,
                self.center_y - self.inner_radius,
                self.center_x + self.inner_radius,
                self.center_y + self.inner_radius,
                start=segment['start'],
                extent=segment_angle,
                fill='#0f172a',
                outline='',
                tags='donut'
            )
            
            drawn_angle += segment_angle
        
        # Texte central
        if self.current_angle >= self.target_angle - 1:
            self.create_text(
                self.center_x,
                self.center_y,
                text="100%",
                fill='#60a5fa',
                font=('Helvetica', 24, 'bold'),
                tags='labels'
            )