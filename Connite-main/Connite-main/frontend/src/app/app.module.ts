import {NoopAnimationsModule} from '@angular/platform-browser/animations';
import {MatToolbarModule} from '@angular/material/toolbar';
import {MatButtonModule} from '@angular/material/button';
import {MatCardModule} from '@angular/material/card';
import {MatInputModule} from '@angular/material/input';

import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';
import {HttpClientModule} from '@angular/common/http';
import {RouterModule, Routes} from '@angular/router';

import {AppComponent} from './app.component';
import {AboutComponent} from './about.component';
import {MatchsApiService} from './matches/matchs-api.service';
import {MatchComponent} from './matches/matchs.component';
import { EndSeasonComponent } from './endseason/endseason.component';

import {Component, ViewEncapsulation} from '@angular/core';
import { DatePipe } from '@angular/common';
import { EndSeasonApiService } from './endseason/endseason-api.service';
import { MatchdataComponent } from './matchdata/matchdata.component';
import { MatTableModule } from '@angular/material/table';
import { StatComponent } from './stat/stat.component' 
import { StatApiService } from './stat/stat-api.service'; 
import { DataApiService } from './matchdata/matchdata-api.service';




const appRoutes: Routes = [
    {path: '', component: MatchComponent},
    {path: 'about', component: AboutComponent},
    {path: 'endSeasonPrediction',component: EndSeasonComponent},
    {path: 'match/:team1/:team2', component: MatchdataComponent},
    {path: 'endSeasonPrediction/:team', component: StatComponent}
];


@NgModule({
  declarations: [
      AppComponent,
      MatchComponent,
      AboutComponent,
      EndSeasonComponent,
      MatchdataComponent,
      StatComponent,
  ],
  imports: [
      BrowserModule,
      HttpClientModule,
      RouterModule.forRoot(appRoutes,),
      NoopAnimationsModule,
      MatToolbarModule,
      MatButtonModule,
      MatCardModule,
      MatInputModule,
      MatTableModule,
      
  ],
    providers: [
        MatchsApiService,
        DatePipe,
        EndSeasonApiService,
        StatApiService,
        DataApiService
        ],
    bootstrap: [AppComponent]
})
export class AppModule { }
