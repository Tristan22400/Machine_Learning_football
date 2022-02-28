import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import Chart from 'chart.js/auto';
import { Rankings } from './rankings.model';
import { StatApiService } from './stat-api.service';
import {Subscription} from 'rxjs';


@Component({
  selector: 'app-stat',
  templateUrl: './stat.component.html',
  styleUrls: ['./stat.component.css']
})
export class StatComponent implements OnInit {
  
  team : string;
  rankingsListSub: Subscription;
  rankings= new Rankings('',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
  myChart : Chart;


  constructor(private route: ActivatedRoute, private StatApi: StatApiService) { }

  ngOnInit(): void {    
      this.route.queryParams.subscribe(params => {
        this.team = this.route.snapshot.paramMap.get('team')
      });
      
      this.rankingsListSub = this.StatApi
      .getRankings(this.team)
      .subscribe(result => 
        {this.rankings = result[0],
           this.myChart = new Chart("myChart", {
            type: 'bar',
            data: {
                labels: ['1', '2', '3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'],
                datasets: [{
                    label: '',
                    data: [this.rankings.One,this.rankings.Two,this.rankings.Three,this.rankings.Four,
                          this.rankings.Five,this.rankings.Six,this.rankings.Seven,this.rankings.Eight,
                          this.rankings.Nine,this.rankings.Ten,this.rankings.Eleven,this.rankings.Twelve,
                          this.rankings.Thirteen,this.rankings.Fourteen,this.rankings.Fifteen,
                          this.rankings.Sixteen,this.rankings.Seventeen,this.rankings.Eighteen,
                          this.rankings.Nineteen,this.rankings.Twenty],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(240,248,255, 0.2)',
                        'rgba(218, 165, 32, 0.2)',
                        'rgba(240,248,255, 0.2)',
                        'rgba(139, 0, 139, 0.2)',
                        'rgba(143, 188, 143, 0.2)',
                        'rgba(139, 69, 19, 0.2)',
                        'rgba(139, 69, 19, 0.2)',
                        'rgba(219, 112, 147, 0.2)',
                        'rgba(255, 228, 181, 0.2)',
                        'rgba(60, 179, 113, 0.2)',
                        'rgba(255, 159, 64, 0.2)',
                        'rgba(0, 0, 205, 0.2)',
                        'rgba(255, 160, 122, 0.2)',
                        'rgba(173, 216, 230, 0.2)',
                        'rgba(255, 255, 240, 0.2)',
                        
                    ],
                    borderColor: [
                      'rgba(255, 99, 132, 1)',
                      'rgba(54, 162, 235, 1)',
                      'rgba(255, 206, 86, 1)',
                      'rgba(75, 192, 192, 1)',
                      'rgba(153, 102, 255, 1)',
                      'rgba(240,248,255, 1)',
                      'rgba(218, 165, 32, 1)',
                      'rgba(240,248,255, 1)',
                      'rgba(139, 0, 139, 1)',
                      'rgba(143, 188, 143,1)',
                      'rgba(139, 69, 19, 1)',
                      'rgba(139, 69, 19, 1)',
                      'rgba(219, 112, 147,1)',
                      'rgba(255, 228, 181, 1)',
                      'rgba(60, 179, 113, 1)',
                      'rgba(255, 159, 64, 1)',
                      'rgba(0, 0, 205, 1)',
                      'rgba(255, 160, 122, 1)',
                      'rgba(173, 216, 230, 1)',
                      'rgba(255, 255, 240, 1)',
                    ],
                    borderWidth: 1
                }]
            },
            options: {
              plugins: {
                legend: {
                  display: false
                },
                title: {
                  display: true,
                  text: 'Rankings Results for ' + this.rankings.Team + ' over 10 000 simulation.'
              }
              },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
      
        
        
        
        }, console.error);




}

}
